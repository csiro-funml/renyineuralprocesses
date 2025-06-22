"""Module for convolutional [conditional | latent] neural processes"""
import logging
import math
import abc
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .new_convnp_utils import CNN, ResConvBlock, SetConv, discard_ith_arg, MLP
from .new_convnp_utils import collapse_z_samples_batch, pool_and_replicate_middle, replicate_z_samples, merge_flat_input, \
    channels_to_2nd_dim, make_abs_conv, channels_to_last_dim, grid_to_number_points
from .new_convnp_utils import MultivariateNormalDiag, isin_range
from .new_convnp_utils import weights_init

logger = logging.getLogger(__name__)

__all__ = ["ConvCNP", "ConvNP", "GridConvNP"]


"""Module for base of [conditional | latent] neural processes"""



class NeuralProcessFamily(nn.Module, abc.ABC):
    """
    Base class for members of the neural process family.

    Notes
    -----
    - when writing size of vectors something like `size=[batch_size,*n_cntxt,y_dim]` means that the
    first dimension is the batch, the last is the target values and everything in the middle are context
    points. We use `*n_cntxt` as it can  be a single flattened dimension or many (for example on the grid).

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    encoded_path : {"latent", "both", "deterministic"}
        Which path(s) to use:
        - `"deterministic"` no latents : the decoder gets a deterministic representation s input.
        - `"latent"` uses latent : the decoder gets a sample latent representation as input.
        - `"both"` concatenates both the deterministic and sampled latents as input to the decoder.

    r_dim : int, optional
        Dimension of representations.

    x_transf_dim : int, optional
        Dimension of the encoded X. If `-1` uses `r_dim`. if `None` uses `x_dim`.

    is_heteroskedastic : bool, optional
        Whether the posterior predictive std can depend on the target features. If using in conjuction
        to `NllLNPF`, it might revert to a *CNP model (collapse of latents). If the flag is False, it
        pools all the scale parameters of the posterior distribution. This trick is only exactly
        recovers heteroskedasticity when the set target features are always the same (e.g.
        predicting values on a predefined grid) but is a good approximation even when not.

    XEncoder : nn.Module, optional
        Spatial encoder module which maps {x^i}_i -> {x_trnsf^i}_i. It should be
        constructable via `XEncoder(x_dim, x_transf_dim)`. `None` uses MLP. Example:
            - `MLP` : will learn positional embeddings with MLP
            - `SinusoidalEncodings` : use sinusoidal positional encodings.

    Decoder : nn.Module, optional
        Decoder module which maps {(x^t, r^t)}_t -> {p_y_suffstat^t}_t. It should be constructable
        via `decoder(x_dim, r_dim, n_out)`. If you have an decoder that maps
        [r;x] -> y you can convert it via `merge_flat_input(Decoder)`. `None` uses MLP. In the
        computational model this corresponds to `g`.
        Example:
            - `merge_flat_input(MLP)` : predict with MLP.
            - `merge_flat_input(SelfAttention, is_sum_merge=True)` : predict
            with self attention mechanisms (using `X_transf + Y` as input) to have
            coherent predictions (not use in attentive neural process [1] but in
            image transformer [2]).
            - `discard_ith_arg(MLP, 0)` if want the decoding to only depend on r.

    PredictiveDistribution : torch.distributions.Distribution, optional
        Predictive distribution. The input to the constructor are currently two values of the same
        shape : `loc` and `scale`, that are preprocessed by `p_y_loc_transformer` and
        `pred_scale_transformer`.

    p_y_loc_transformer : callable, optional
        Transformation to apply to the predicted location (e.g. mean for Gaussian)
        of Y_trgt.

    p_y_scale_transformer : callable, optional
        Transformation to apply to the predicted scale (e.g. std for Gaussian) of
        Y_trgt. The default follows [3] by using a minimum of 0.01.

    References
    ----------
    [1] Kim, Hyunjik, et al. "Attentive neural processes." arXiv preprint
        arXiv:1901.05761 (2019).
    [2] Parmar, Niki, et al. "Image transformer." arXiv preprint arXiv:1802.05751
        (2018).
    [3] Le, Tuan Anh, et al. "Empirical Evaluation of Neural Process Objectives."
        NeurIPS workshop on Bayesian Deep Learning. 2018.
    """

    _valid_paths = ["deterministic", "latent", "both"]

    def __init__(
        self,
        x_dim,
        y_dim,
        encoded_path,
        r_dim=128,
        x_transf_dim=-1,
        is_heteroskedastic=True,
        XEncoder=None,
        Decoder=None,
        PredictiveDistribution=MultivariateNormalDiag,
        p_y_loc_transformer=nn.Identity(),
        p_y_scale_transformer=lambda y_scale: 0.01 + 0.99 * F.softplus(y_scale),
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.encoded_path = encoded_path
        self.is_heteroskedastic = is_heteroskedastic

        if x_transf_dim is None:
            self.x_transf_dim = self.x_dim
        elif x_transf_dim == -1:
            self.x_transf_dim = self.r_dim
        else:
            self.x_transf_dim = x_transf_dim

        self.encoded_path = encoded_path.lower()
        if self.encoded_path not in self._valid_paths:
            raise ValueError(f"Unknown encoded_path={self.encoded_path}.")

        if XEncoder is None:
            XEncoder = self.dflt_Modules["XEncoder"]

        if Decoder is None:
            Decoder = self.dflt_Modules["Decoder"]

        self.x_encoder = XEncoder(self.x_dim, self.x_transf_dim)

        # times 2 out because loc and scale (mean and var for gaussian)
        self.decoder = Decoder(self.x_transf_dim, self.r_dim, self.y_dim * 2)

        self.PredictiveDistribution = PredictiveDistribution
        self.p_y_loc_transformer = p_y_loc_transformer
        self.p_y_scale_transformer = p_y_scale_transformer

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    @property
    def dflt_Modules(self):
        dflt_Modules = dict()

        dflt_Modules["XEncoder"] = partial(
            MLP, n_hidden_layers=1, hidden_size=self.r_dim
        )

        dflt_Modules["SubDecoder"] = partial(
            MLP,
            n_hidden_layers=4,
            hidden_size=self.r_dim,
        )

        dflt_Modules["Decoder"] = merge_flat_input(
            dflt_Modules["SubDecoder"], is_sum_merge=True
        )

        return dflt_Modules

    def forward(self,batch, num_samples = None, reduce_ll = True):
        """
        Given a set of context feature-values {(x^c, y^c)}_c and target features {x^t}_t, return
        a set of posterior distribution for target values {p(Y^t|y_c; x_c, x_t)}_t.

        Parameters
        ----------
        X_cntxt: torch.Tensor, size=[batch_size, *n_cntxt, x_dim]
            Set of all context features {x_i}. Values need to be in interval [-1,1]^d.

        Y_cntxt: torch.Tensor, size=[batch_size, *n_cntxt, y_dim]
            Set of all context values {y_i}.

        X_trgt: torch.Tensor, size=[batch_size, *n_trgt, x_dim]
            Set of all target features {x_t}. Values need to be in interval [-1,1]^d.

        Y_trgt: torch.Tensor, size=[batch_size, *n_trgt, y_dim], optional
            Set of all target values {y_t}. Only required during training and if
            using latent path.

        Return
        ------
        p_y_trgt: torch.distributions.Distribution, batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
            Posterior distribution for target values {p(Y^t|y_c; x_c, x_t)}_t

        z_samples: torch.Tensor, size=[n_z_samples, batch_size, *n_lat, r_dim]
            Sampled latents. `None` if `encoded_path==deterministic`.

        q_zCc: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[r_dim]
            Latent distribution for the context points. `None` if `encoded_path==deterministic`.

        q_zCct: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[r_dim]
            Latent distribution for the targets. `None` if `encoded_path==deterministic`
            or not training or not `is_q_zCct`.
        """
        X_cntxt, Y_cntxt, X_trgt, Y_trgt = batch.xc, batch.yc, batch.x, batch.y
        if not self.training:
            Y_trgt = None
        self._validate_inputs(X_cntxt, Y_cntxt, X_trgt, Y_trgt)

        # size = [batch_size, *n_cntxt, x_transf_dim]
        X_cntxt = self.x_encoder(X_cntxt)
        # size = [batch_size, *n_trgt, x_transf_dim]
        X_trgt = self.x_encoder(X_trgt)

        # {R^u}_u
        # size = [batch_size, *n_rep, r_dim]
        R = self.encode_globally(X_cntxt, Y_cntxt)

        if self.encoded_path in ["latent", "both"]:
            z_samples, q_zCc, q_zCct = self.latent_path(X_cntxt, R, X_trgt, Y_trgt)
        else:
            z_samples, q_zCc, q_zCct = None, None, None

        if self.encoded_path == "latent":
            # if only latent path then cannot depend on deterministic representation
            R = None

        # size = [n_z_samples, batch_size, *n_trgt, r_dim]
        R_trgt = self.trgt_dependent_representation(X_cntxt, z_samples, R, X_trgt)

        # p(y|cntxt,trgt)
        # batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
        p_yCc = self.decode(X_trgt, R_trgt)

        # if 'mask_xc' in batch.keys(): # ConvGridNP
            # reshape them into (n_z, bs, n_points, dim)
            # if q_zCc is not None:
            #     q_zCc = Normal(grid_to_number_points(q_zCc.loc), grid_to_number_points(q_zCc.scale))
            # if q_zCct is not None:
            #     q_zCct = Normal(grid_to_number_points(q_zCct.loc), grid_to_number_points(q_zCct.scale))
            # z_samples = grid_to_number_points(z_samples)
            # p_yCc = Normal(grid_to_number_points(p_yCc.loc), grid_to_number_points(p_yCc.scale))
        return p_yCc, z_samples, q_zCc, q_zCct

    def _validate_inputs(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt):
        """Validates the inputs by checking if features are rescaled to [-1,1] during training."""
        if self.training:
            if not (isin_range(X_cntxt, [-2, 2]) and isin_range(X_trgt, [-2, 2])):
                raise ValueError(
                    f"Features during training should be in [-2,2]. {X_cntxt.min()} <= X_cntxt <= {X_cntxt.max()} ; {X_trgt.min()} <= X_trgt <= {X_trgt.max()}."
                )

    @abc.abstractmethod
    def encode_globally(self, X_cntxt, R_cntxt):
        """Encode context set all together (globally).

        Parameters
        ----------
        X_cntxt : torch.Tensor, size=[batch_size, *n_cntxt, x_transf_dim]
            Set of all context features {x^c}_c.

        Y_cntxt: torch.Tensor, size=[batch_size, *n_cntxt, y_dim]
            Set of all context values {y^c}_c.

        Return
        ------
        R : torch.Tensor, size=[batch_size, *n_rep, r_dim]
            Global representations of the context set.
        """
        pass

    @abc.abstractmethod
    def trgt_dependent_representation(self, X_cntxt, z_samples, R, X_trgt):
        """Compute a target dependent representation of the context set.

        Parameters
        ----------
        X_cntxt : torch.Tensor, size=[batch_size, *n_cntxt, x_transf_dim]
            Set of all context features {x^c}_c.

        z_samples: torch.Tensor, size=[n_z_samples, batch_size, *n_lat, r_dim]
            Sampled latents. `None` if `encoded_path==deterministic`.

        R : torch.Tensor, size=[batch_size, *n_rep, r_dim]
            Global representation of the context set. `None` if `self.encoded_path==latent`.

        X_trgt : torch.Tensor, size=[batch_size, *n_trgt, x_transf_dim]
            Set of all target features {x^t}_t.

        Returns
        -------
        R_trgt : torch.Tensor, size=[n_z_samples, batch_size, *n_trgt, r_dim]
            Set of all target representations {r^t}_t.
        """
        pass

    def latent_path(self, X_cntxt, R, X_trgt, Y_trgt):
        """Infer latent variable given context features and global representation.

        Parameters
        ----------
        R : torch.Tensor, size=[batch_size, *n_rep, r_dim]
            Global representation values {r^u}_u.

        X_cntxt : torch.Tensor, size=[batch_size, *n_cntxt, x_transf_dim]
            Set of all context features {x^c}_c.

        X_trgt : torch.Tensor, size=[batch_size, *n_trgt, x_transf_dim]
            Set of all target features {x^t}_t.

        Y_trgt: torch.Tensor, size=[batch_size, *n_trgt, y_dim], optional
            Set of all target values {y_t}. Only required during training and if
            using latent path.

        Return
        ------
        z_samples: torch.Tensor, size=[n_z_samples, batch_size, *n_lat, r_dim]
            Sampled latents. `None` if `encoded_path==deterministic`.

        q_zCc: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[r_dim]
            Latent distribution for the context points. `None` if `encoded_path==deterministic`.

        q_zCct: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[r_dim]
            Latent distribution for the targets. `None` if `encoded_path==deterministic`
            or not training or not `is_q_zCct`.
        """
        raise NotImplementedError(
            f"`latent_path` not implemented. Cannot use encoded_path={self.encoded_path} in such case."
        )

    def decode(self, X_trgt, R_trgt):
        """
        Compute predicted distribution conditioned on representation and
        target positions.

        Parameters
        ----------
        X_trgt: torch.Tensor, size=[batch_size, *n_trgt, x_transf_dim]
            Set of all target features {x^t}_t.

        R_trgt : torch.Tensor, size=[n_z_samples, batch_size, *n_trgt, r_dim]
            Set of all target representations {r^t}_t.

        Return
        ------
        p_y_trgt: torch.distributions.Distribution, batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
            Posterior distribution for target values {p(Y^t|y_c; x_c, x_t)}_t

        """
        # size = [n_z_samples, batch_size, *n_trgt, y_dim*2]
        p_y_suffstat = self.decoder(X_trgt, R_trgt)

        # size = [n_z_samples, batch_size, *n_trgt, y_dim]
        p_y_loc, p_y_scale = p_y_suffstat.split(self.y_dim, dim=-1)

        p_y_loc = self.p_y_loc_transformer(p_y_loc)
        p_y_scale = self.p_y_scale_transformer(p_y_scale)

        #! shuld probably pool before p_y_scale_transformer
        if not self.is_heteroskedastic:
            # to make sure not heteroskedastic you pool all the p_y_scale
            # only exact when X_trgt is a constant (e.g. grid case). If not it's a descent approx
            n_z_samples, batch_size, *n_trgt, y_dim = p_y_scale.shape
            p_y_scale = p_y_scale.view(n_z_samples * batch_size, *n_trgt, y_dim)
            p_y_scale = pool_and_replicate_middle(p_y_scale)
            p_y_scale = p_y_scale.view(n_z_samples, batch_size, *n_trgt, y_dim)

        # batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
        p_yCc = self.PredictiveDistribution(p_y_loc, p_y_scale)

        return p_yCc

    def set_extrapolation(self, min_max):
        """Set the neural process for extrapolation."""
        pass


class LatentNeuralProcessFamily(NeuralProcessFamily):
    """Base class for members of the latent neural process (sub-)family.

    Parameters
    ----------
    *args:
        Positional arguments to `NeuralProcessFamily`.

    encoded_path : {"latent", "both"}
        Which path(s) to use:
        - `"latent"` uses latent : the decoder gets a sample latent representation as input.
        - `"both"` concatenates both the deterministic and sampled latents as input to the decoder.

    is_q_zCct : bool, optional
        Whether to infer Z using q(Z|cntxt,trgt) instead of q(Z|cntxt). This requires the loss
        to perform some type of importance sampling. Only used if `encoded_path in {"latent", "both"}`.

    n_z_samples_train : int or scipy.stats.rv_frozen, optional
        Number of samples from the latent during training. Only used if `encoded_path in {"latent", "both"}`.
        Can also be a scipy random variable , which is useful if the number of samples has to be stochastic, for
        example when using `SUMOLossNPF`.

    n_z_samples_test : int or scipy.stats.rv_frozen, optional
        Number of samples from the latent during testing. Only used if `encoded_path in {"latent", "both"}`.
        Can also be a scipy random variable , which is useful if the number of samples has to be stochastic, for
        example when using `SUMOLossNPF`.

    LatentEncoder : nn.Module, optional
        Encoder which maps r -> z_suffstat. It should be constructed via
        `LatentEncoder(r_dim, n_out)`.  If `None` uses an MLP.

    LatentDistribution : torch.distributions.Distribution, optional
        Latent distribution. The input to the constructor are currently two values  : `loc` and `scale`,
        that are preprocessed by `q_z_loc_transformer` and `q_z_loc_transformer`.

    q_z_loc_transformer : callable, optional
        Transformation to apply to the predicted location (e.g. mean for Gaussian)
        of Y_trgt.

    q_z_scale_transformer : callable, optional
        Transformation to apply to the predicted scale (e.g. std for Gaussian) of
        Y_trgt. The default follows [3] by using a minimum of 0.1 and maximum of 1.

    **kwargs:
        Additional arguments to `NeuralProcessFamily`.
    """

    _valid_paths = ["latent", "both"]

    def __init__(
        self,
        *args,
        is_q_zCct=False,
        n_z_samples_train=32,
        n_z_samples_test=32,
        LatentEncoder=None,
        LatentDistribution=MultivariateNormalDiag,
        q_z_loc_transformer=nn.Identity(),
        q_z_scale_transformer=lambda z_scale: 0.1 + 0.9 * torch.sigmoid(z_scale),
        z_dim=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.is_q_zCct = is_q_zCct
        self.n_z_samples_train = n_z_samples_train
        self.n_z_samples_test = n_z_samples_test
        self.z_dim = self.r_dim if z_dim is None else z_dim

        if LatentEncoder is None:
            LatentEncoder = self.dflt_Modules["LatentEncoder"]

        # times 2 out because loc and scale (mean and var for gaussian)
        self.latent_encoder = LatentEncoder(self.r_dim, self.z_dim * 2)

        if self.encoded_path == "both":
            self.r_z_merger = nn.Linear(self.r_dim + self.z_dim, self.r_dim)

        self.LatentDistribution = LatentDistribution
        self.q_z_loc_transformer = q_z_loc_transformer
        self.q_z_scale_transformer = q_z_scale_transformer

        if self.z_dim != self.r_dim and self.encoded_path == "latent":
            # will reshape the z samples to make sure they can be given to the decoder
            self.reshaper_z = nn.Linear(self.z_dim, self.r_dim)

        self.reset_parameters()

    @property
    def dflt_Modules(self):
        # allow inheritence
        dflt_Modules = NeuralProcessFamily.dflt_Modules.__get__(self)

        dflt_Modules["LatentEncoder"] = partial(
            MLP,
            n_hidden_layers=1,
            hidden_size=self.r_dim,
        )

        return dflt_Modules

    def forward(self, *args, **kwargs):

        # make sure that only sampling oce per loop => cannot be a property
        try:
            # if scipy random variable, i.e., random number of samples
            self.n_z_samples = (
                self.n_z_samples_train.rvs()
                if self.training
                else self.n_z_samples_test.rvs()
            )
        except AttributeError:
            self.n_z_samples = (
                self.n_z_samples_train if self.training else self.n_z_samples_test
            )

        return super().forward(*args, **kwargs)

    def _validate_inputs(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt):
        super()._validate_inputs(X_cntxt, Y_cntxt, X_trgt, Y_trgt)

    def latent_path(self, X_cntxt, R, X_trgt, Y_trgt):
        """

        Args:
            X_cntxt:
            R:
            X_trgt:
            Y_trgt:

        Returns:
            q_zCc is the context distribution
            q_zCct is the target distribution
        """

        # q(z|c)
        # batch shape = [batch_size, *n_lat] ; event shape = [z_dim]
        q_zCc = self.infer_latent_dist(X_cntxt, R)

        if self.is_q_zCct and Y_trgt is not None:
            # during training when we know Y_trgt, we can take an expectation over q(z|cntxt,trgt)
            # instead of q(z|cntxt). note that actually does q(z|trgt) because trgt has cntxt
            R_from_trgt = self.encode_globally(X_trgt, Y_trgt)
            q_zCct = self.infer_latent_dist(X_trgt, R_from_trgt)
            sampling_dist = q_zCct
        else:
            q_zCct = None
            sampling_dist = q_zCc

        # size = [n_z_samples, batch_size, *n_lat, z_dim]
        z_samples = sampling_dist.rsample([self.n_z_samples])

        return z_samples, q_zCc, q_zCct

    def infer_latent_dist(self, X, R):
        """Infer latent distribution given desired features and global representation.

        Parameters
        ----------
        X : torch.Tensor, size=[batch_size, *n_i, x_transf_dim]
            Set of all features {x^i}_i. E.g. context or target.

        R : torch.Tensor, size=[batch_size, *n_rep, r_dim]
            Global representation values {r^u}_u.

        Return
        ------
        q_zCc: torch.distributions.Distribution, batch shape = [batch_size, *n_lat] ; event shape = [z_dim]
            Inferred latent distribution.
        """

        # size = [batch_size, *n_lat, z_dim]
        R_lat_inp = self.rep_to_lat_input(R)

        # size = [batch_size, *n_lat, z_dim*2]
        q_z_suffstat = self.latent_encoder(R_lat_inp)

        q_z_loc, q_z_scale = q_z_suffstat.split(self.z_dim, dim=-1)

        q_z_loc = self.q_z_loc_transformer(q_z_loc)
        q_z_scale = self.q_z_scale_transformer(q_z_scale)

        # batch shape = [batch_size, *n_lat] ; event shape = [z_dim]
        q_zCc = self.LatentDistribution(q_z_loc, q_z_scale)

        return q_zCc

    def rep_to_lat_input(self, R):
        """Transform the n_rep representations to n_lat inputs."""
        # by default *n_rep = *n_lat
        return R

    def merge_r_z(self, R, z_samples):
        """
        Merges the deterministic representation and sampled latent. Assumes that n_lat = n_rep.

        Parameters
        ----------
        R : torch.Tensor, size=[batch_size, *, r_dim]
            Global representation values {r^u}_u.

        z_samples : torch.Tensor, size=[n_z_samples, batch_size, *, r_dim]
            Global representation values {r^u}_u.

        Return
        ------
        out : torch.Tensor, size=[n_z_samples, batch_size, *, r_dim]
        """
        if R.shape != z_samples.shape:

            R = R.unsqueeze(0).expand(*z_samples.shape[:-1], self.r_dim)

        # (add ReLU to not have linear followed by linear)
        return torch.relu(self.r_z_merger(torch.cat((R, z_samples), dim=-1)))



class ConvCNP(NeuralProcessFamily):
    """
    Convolutional conditional neural process [1].

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    density_induced : int, optional
        Density of induced-inputs to use. The induced-inputs will be regularly sampled.

    Interpolator : callable or str, optional
        Callable to use to compute cntxt / trgt to and from the induced points.  {(x^k, y^k)}, {x^q} -> {y^q}.
        It should be constructed via `Interpolator(x_dim, in_dim, out_dim)`. Example:
            - `SetConv` : uses a set convolution as in the paper.
            - `"TransformerAttender"` : uses a cross attention layer.

    CNN : nn.Module, optional
        Convolutional model to use between induced points. It should be constructed via
        `CNN(r_dim)`. Important : the channel needs to be last dimension of input. Example:
            - `partial(CNN,ConvBlock=ResConvBlock,Conv=nn.Conv2d,is_chan_last=True` : uses a small
            ResNet.
            - `partial(UnetCNN,ConvBlock=ResConvBlock,Conv=nn.Conv2d,is_chan_last=True` : uses a
            UNet.

    kwargs :
        Additional arguments to `NeuralProcessFamily`.

    References
    ----------
    [1] Gordon, Jonathan, et al. "Convolutional conditional neural processes." arXiv preprint
    arXiv:1910.13556 (2019).
    """

    _valid_paths = ["deterministic"]

    def __init__(
        self,
        x_dim,
        y_dim,
        density_induced=128,
        Interpolator=SetConv,
        CNN=partial(
            CNN,
            ConvBlock=ResConvBlock,
            Conv=nn.Conv1d,
            n_blocks=3,
            Normalization=nn.Identity,
            is_chan_last=True,
            kernel_size=11,
        ),
        **kwargs,
    ):

        if (
            "Decoder" in kwargs and kwargs["Decoder"] != nn.Identity
        ):  # identity means that not using
            logger.warning(
                "`Decoder` was given to `ConvCNP`. To be translation equivariant you should disregard the first argument for example using `discard_ith_arg(Decoder, i=0)`, which is done by default when you DO NOT provide the Decoder."
            )

        # don't force det so that can inherit ,
        kwargs["encoded_path"] = kwargs.get("encoded_path", "deterministic")
        super().__init__(
            x_dim,
            y_dim,
            x_transf_dim=None,
            XEncoder=nn.Identity,
            **kwargs,
        )

        self.density_induced = density_induced
        # input is between -1 and 1 but use at least 0.5 temporary values on each sides to not
        # have strong boundary effects
        self.X_induced = torch.linspace(-1.5, 1.5, int(self.density_induced * 3))
        self.CNN = CNN

        self.cntxt_to_induced = Interpolator(self.x_dim, self.y_dim, self.r_dim)
        self.induced_to_induced = CNN(self.r_dim)
        self.induced_to_trgt = Interpolator(self.x_dim, self.r_dim, self.r_dim)

        self.reset_parameters()

    @property
    def n_induced(self):
        # using property because this might change after you set extrapolation
        return len(self.X_induced)

    @property
    def dflt_Modules(self):
        # allow inheritence
        dflt_Modules = NeuralProcessFamily.dflt_Modules.__get__(self)

        # don't depend on x
        dflt_Modules["Decoder"] = discard_ith_arg(dflt_Modules["SubDecoder"], i=0)

        return dflt_Modules

    def _get_X_induced(self, X):
        batch_size, _, _ = X.shape

        # effectively puts on cuda only once
        self.X_induced = self.X_induced.to(X.device)
        X_induced = self.X_induced.view(1, -1, 1)
        X_induced = X_induced.expand(batch_size, self.n_induced, self.x_dim)
        return X_induced

    def encode_globally(self, X_cntxt, Y_cntxt):
        batch_size, n_cntxt, _ = X_cntxt.shape

        # size = [batch_size, n_induced, x_dim]
        X_induced = self._get_X_induced(X_cntxt)

        # size = [batch_size, n_induced, r_dim]
        R_induced = self.cntxt_to_induced(X_cntxt, X_induced, Y_cntxt)

        if n_cntxt == 0:
            # arbitrarily setting the global representation to zero when no context
            # but the density channel will also be => makes sense
            R_induced = torch.zeros(
                batch_size, self.n_induced, self.r_dim, device=R_induced.device
            )

        # size = [batch_size, n_induced, r_dim]
        R_induced = self.induced_to_induced(R_induced)

        return R_induced

    def trgt_dependent_representation(self, X_cntxt, z_samples, R_induced, X_trgt):
        batch_size, n_trgt, _ = X_trgt.shape

        # size = [batch_size, n_induced, x_dim]
        X_induced = self._get_X_induced(X_cntxt)

        # size = [batch_size, n_trgt, r_dim]
        R_trgt = self.induced_to_trgt(X_induced, X_trgt, R_induced)

        # n_z_samples=1. size = [1, batch_size, n_trgt, r_dim]
        return R_trgt.unsqueeze(0)

    def set_extrapolation(self, min_max):
        """
        Scale the induced inputs to be in a given range while keeping
        the same density than during training (used for extrapolation.).
        """
        current_min = min_max[0] - 0.5
        current_max = min_max[1] + 0.5
        self.X_induced = torch.linspace(
            current_min,
            current_max,
            int(self.density_induced * (current_max - current_min)),
        )


class ConvNP(LatentNeuralProcessFamily, ConvCNP):
    """
    Convolutional latent neural process [1].

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    is_global : bool, optional
        Whether to also use a global representation in addition to the latent one. Only if
        encoded_path = `latent`.

    CNNPostZ : Module, optional
        CNN to use after the sampling. If `None` uses the same as before sampling. Note that computations
        will be heavier after sampling (as performing on all the samples) so you might want to
        make it smaller.

    kwargs :
        Additional arguments to `ConvCNP`.

    References
    ----------
    [1] Foong, Andrew YK, et al. "Meta-Learning Stationary Stochastic Process Prediction with
    Convolutional Neural Processes." arXiv preprint arXiv:2007.01332 (2020).
    """

    _valid_paths = ["latent", "both"]

    def __init__(
        self,
        x_dim,
        y_dim,
        CNNPostZ=None,
        encoded_path="latent",
        is_global=False,
        **kwargs,
    ):
        super().__init__(
            x_dim,
            y_dim,
            encoded_path=encoded_path,
            **kwargs,
        )

        self.is_global = is_global

        if CNNPostZ is None:
            CNNPostZ = self.CNN

        self.induced_to_induced_post_sampling = CNNPostZ(self.r_dim)

        self.reset_parameters()

    @property
    def dflt_Modules(self):
        # allow inheritance
        dflt_Modules = ConvCNP.dflt_Modules.__get__(self)
        dflt_Modules2 = LatentNeuralProcessFamily.dflt_Modules.__get__(self)
        dflt_Modules.update(dflt_Modules2)

        # use smaller decoder than ConvCNP because more param due to `induced_to_induced_post_sampling`
        dflt_Modules["Decoder"] = discard_ith_arg(nn.Linear, i=0)

        return dflt_Modules

    def rep_to_lat_input(self, R):
        batch_size, *n_induced, _ = R.shape

        if self.encoded_path == "latent":
            # size = [batch_size, *n_induced, r_dim]
            return R

        elif self.encoded_path == "both":
            # size = [batch_size, 1, r_dim]
            return R.view(batch_size, -1, self.r_dim).mean(dim=1, keepdim=True)

    def trgt_dependent_representation(self, X_cntxt, z_samples, R_induced, X_trgt):

        batch_size, n_trgt, _ = X_trgt.shape
        n_z_samples, _, n_lat, __ = z_samples.shape

        # size = [batch_size, n_induced, x_dim]
        X_induced = self._get_X_induced(X_cntxt)

        # size = [n_z_samples*batch_size, *, x_dim]
        X_induced = collapse_z_samples_batch(
            replicate_z_samples(X_induced, n_z_samples)
        )
        X_trgt = collapse_z_samples_batch(replicate_z_samples(X_trgt, n_z_samples))

        if self.encoded_path == "latent":
            # make all computations with n_z_samples and batch size merged (because CNN need it)
            # size = [n_z_samples * batch_size, n_induced, z_dim]
            z_samples = collapse_z_samples_batch(z_samples)

            # size = [n_z_samples * batch_size, n_induced, r_dim]
            if self.z_dim != self.r_dim:
                z_samples = self.reshaper_z(z_samples)

            # size = [n_z_samples*batch_size, n_induced, r_dim]
            # "mixing" after the sampling to have coherent samples
            z_samples = self.induced_to_induced_post_sampling(z_samples)

            #! SHOULD be directly after sampling (like in gridconvnp)
            if self.is_global:
                # size = [n_z_samples*batch_size, n_induced, r_dim]
                z_samples = self.add_global_latent(z_samples)

            # size = [n_z_samples * batch_size, n_trgt, r_dim]
            R_trgt = self.induced_to_trgt(X_induced, X_trgt, z_samples)

        elif self.encoded_path == "both":
            # size = [n_z_samples, batch_size, n_induced, z_dim]
            z_samples = z_samples.expand(
                n_z_samples, batch_size, self.n_induced, self.z_dim
            )

            R_induced = self.merge_r_z(R_induced, z_samples)

            # make all computations with n_z_samples and batch size merged (because CNN need it)
            # size = [n_z_samples * batch_size, self.n_induced, self.r_dim]
            R_induced = collapse_z_samples_batch(R_induced)

            # to make it comparable with `latent` path
            R_induced = self.induced_to_induced_post_sampling(R_induced)

            # size = [n_z_samples * batch_size, n_trgt, r_dim]
            R_trgt = self.induced_to_trgt(X_induced, X_trgt, R_induced)

        # extracts n_z_dim
        R_trgt = R_trgt.view(n_z_samples, batch_size, n_trgt, self.r_dim)

        return R_trgt

    def add_global_latent(self, z_samples):
        """Add a global latent to z_samples."""
        # size = [n_z_samples*batch_size, n_induced, r_dim // 2]
        local_z_samples, global_z_samples = z_samples.split(
            z_samples.shape[-1] // 2, dim=-1
        )

        # size = [n_z_samples*batch_size, n_induced, r_dim //2]
        global_z_samples = pool_and_replicate_middle(global_z_samples)

        # size = [n_z_samples*batch_size, n_induced * 2, r_dim]
        z_samples = torch.cat([local_z_samples, global_z_samples], dim=-1)

        return z_samples



class GridConvCNP(NeuralProcessFamily):
    """
    Spacial case of Convolutional Conditional Neural Process [1] when the context, targets and
    induced points points are on a grid of the same size.

    Notes
    -----
    - Assumes that input, output and induced points are on the same grid. I.e. This cannot be used
    for sub-pixel interpolation / super resolution. I.e. in the code *n_rep = *n_cntxt = *n_trgt =* grid_shape.
    The real number of ontext and target will be determined by the masks.
    - Assumes that Y_cntxt is the grid values (y_dim / channels on last dim),
    while X_cntxt and X_trgt are confidence masks of the shape of the grid rather
    than set of features.
    - As X_cntxt and X_trgt is a grid, each batch example could have a different number of
    contexts  and targets (i.e. different number of non zeros).
    - As we do not use a set convolution, the receptive field is easy to specify,
    making the model much more computationally efficient.

    Parameters
    ----------
    x_dim : int
        Dimension of features. As the features are now masks, this has to be either 1 or y_dim
        as they will be multiplied to Y (with possible broadcasting). If 1 then selectign all channels
        or none.

    y_dim : int
        Dimension of y values.

    Conv : nn.Module, optional
        Convolution layer to use to map from context to induced points {(x^k, y^k)}, {x^q} -> {y^q}.

    CNN : nn.Module, optional
        Convolutional model to use between induced points. It should be constructed via
        `CNN(r_dim)`. Important : the channel needs to be last dimension of input. Example:
            - `partial(CNN,ConvBlock=ResConvBlock,Conv=nn.Conv2d,is_chan_last=True` : uses a small
            ResNet.
            - `partial(UnetCNN,ConvBlock=ResConvBlock,Conv=nn.Conv2d,is_chan_last=True` : uses a
            UNet.

    kwargs :
        Additional arguments to `ConvCNP`.

    References
    ----------
    [1] Gordon, Jonathan, et al. "Convolutional conditional neural processes." arXiv preprint
    arXiv:1910.13556 (2019).
    """

    _valid_paths = ["deterministic"]

    def __init__(
        self,
        x_dim,
        y_dim,
        # uses only depth wise + make sure positive to be interpreted as a density
        Conv=lambda y_dim: make_abs_conv(nn.Conv2d)(
            y_dim,
            y_dim,
            groups=y_dim,
            kernel_size=11,
            padding=11 // 2,
            bias=False,
        ),
        CNN=partial(
            CNN,
            ConvBlock=ResConvBlock,
            Conv=nn.Conv2d,
            n_blocks=3,
            Normalization=nn.Identity,
            is_chan_last=True,
            kernel_size=11,
        ),
        **kwargs,
    ):

        assert (
            x_dim == 1 or x_dim == y_dim
        ), "Ensure that featrue masks can be multiplied with Y"

        if (
            "Decoder" in kwargs and kwargs["Decoder"] != nn.Identity
        ):  # identity means that not using
            logger.warning(
                "`Decoder` was given to `ConvCNP`. To be translation equivariant you should disregard the first argument for example using `discard_ith_arg(Decoder, i=0)`, which is done by default when you DO NOT provide the Decoder."
            )

        # don't force det so that can inherit ,
        kwargs["encoded_path"] = kwargs.get("encoded_path", "deterministic")
        super().__init__(
            x_dim,
            y_dim,
            x_transf_dim=None,
            XEncoder=nn.Identity,
            **kwargs,
        )

        self.CNN = CNN
        self.conv = Conv(y_dim)
        self.resizer = nn.Linear(
            self.y_dim * 2, self.r_dim
        )  # 2 because also confidence channels

        self.induced_to_induced = CNN(self.r_dim)

        self.reset_parameters()

    dflt_Modules = ConvCNP.dflt_Modules

    def cntxt_to_induced(self, mask_cntxt, X):
        """Infer the missing values  and compute a density channel."""

        # channels have to be in second dimension for convolution
        # size = [batch_size, y_dim, *grid_shape]
        X = channels_to_2nd_dim(X)
        # size = [batch_size, x_dim, *grid_shape]
        mask_cntxt = channels_to_2nd_dim(mask_cntxt).float()

        # size = [batch_size, y_dim, *grid_shape]
        X_cntxt = X * mask_cntxt
        signal = self.conv(X_cntxt)
        density = self.conv(mask_cntxt.expand_as(X))

        # normalize
        out = signal / torch.clamp(density, min=1e-5)

        # size = [batch_size, y_dim * 2, *grid_shape]
        out = torch.cat([out, density], dim=1)

        # size = [batch_size, *grid_shape, y_dim * 2]
        out = channels_to_last_dim(out)

        # size = [batch_size, *grid_shape, r_dim]
        out = self.resizer(out)

        return out

    def encode_globally(self, mask_cntxt, X):

        # size = [batch_size, *grid_shape, r_dim]
        R_induced = self.cntxt_to_induced(mask_cntxt, X)
        R_induced = self.induced_to_induced(R_induced)

        return R_induced

    def trgt_dependent_representation(self, _, __, R_induced, ___):

        # n_z_samples=1. size = [1, batch_size, n_trgt, r_dim]
        return R_induced.unsqueeze(0)

    def set_extrapolation(self, min_max):
        raise NotImplementedError("GridConvCNP cannot be used for extrapolation.")


class GridConvNP(LatentNeuralProcessFamily, GridConvCNP):
    """
    Spacial case of Convolutional Latent Neural Process [1] when the context, targets and
    induced points points are on a grid of the same size. C.f. `GridConvCNP` for more details.

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    is_global : bool, optional
        Whether to also use a global representation in addition to the latent one. Only if
        encoded_path = `latent`.

    CNNPostZ : Module, optional
        CNN to use after the sampling. If `None` uses the same as before sampling. Note that computations
        will be heavier after sampling (as performing on all the samples) so you might want to
        make it smaller.

    kwargs :
        Additional arguments to `ConvCNP`.

    References
    ----------
    [1] Gordon, Jonathan, et al. "Convolutional conditional neural processes." arXiv preprint
    arXiv:1910.13556 (2019).
    """

    _valid_paths = ["latent", "both"]

    def __init__(
        self,
        x_dim,
        y_dim,
        CNNPostZ=None,
        encoded_path="latent",
        is_global=False,
        **kwargs,
    ):
        super().__init__(
            x_dim,
            y_dim,
            encoded_path=encoded_path,
            **kwargs,
        )

        self.is_global = is_global

        if CNNPostZ is None:
            CNNPostZ = self.CNN

        self.induced_to_induced_post_sampling = CNNPostZ(self.r_dim)

        self.reset_parameters()

    # maybe should inherit from ConvLNP instead ?
    dflt_Modules = ConvNP.dflt_Modules
    add_global_latent = ConvNP.add_global_latent
    rep_to_lat_input = ConvNP.rep_to_lat_input

    def trgt_dependent_representation(self, X_cntxt, z_samples, R_induced, X_trgt):
        batch_size, *grid_shape, _ = X_trgt.shape
        n_z_samples = z_samples.size(0)

        if self.encoded_path == "latent":

            # make all computations with n_z_samples and batch size merged (because CNN need it)
            # size = [n_z_samples*batch_size, *grid_shape, r_dim]
            z_samples = collapse_z_samples_batch(z_samples)

            if self.is_global:
                # size = [n_z_samples*batch_size, *grid_shape, z_dim]
                z_samples = self.add_global_latent(z_samples)

            # size = [n_z_samples * batch_size, n_induced, r_dim]
            if self.z_dim != self.r_dim:
                z_samples = self.reshaper_z(z_samples)

            # size = [n_z_samples*batch_size, *grid_shape, r_dim]
            # "mixing" after the sampling to have coherent samples
            R_trgt = self.induced_to_induced_post_sampling(z_samples)

        elif self.encoded_path == "both":
            # z_samples is size = [n_z_samples, batch_size, 1, r_dim]
            z_samples = z_samples.view(
                n_z_samples, batch_size, *([1] * len(grid_shape)), self.r_dim
            )
            z_samples = z_samples.expand(
                n_z_samples, batch_size, *grid_shape, self.r_dim
            )

            # size = [n_z_samples, batch_size, *grid_shape, r_dim]
            R_induced = self.merge_r_z(R_induced, z_samples)

            # make all computations with n_z_samples and batch size merged (because CNN need it)
            # size = [n_z_samples*batch_size, *grid_shape, r_dim]
            R_induced = collapse_z_samples_batch(R_induced)

            # size = [n_z_samples*batch_size, *grid_shape, r_dim]
            # to make it comparable with `latent` path
            R_trgt = self.induced_to_induced_post_sampling(R_induced)

        # extracts n_z_dim
        R_trgt = R_trgt.view(n_z_samples, batch_size, *grid_shape, self.r_dim)

        return R_trgt