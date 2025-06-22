import logging
import math
from functools import partial
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import warnings
import operator
from functools import reduce
from torch.distributions.independent import Independent

logger = logging.getLogger(__name__)


def isin_range(x, valid_range):
    """Check if array / tensor is in a given range elementwise."""
    return ((x >= valid_range[0]) & (x <= valid_range[1])).all()


def get_activation_name(activation):
    """Given a string or a `torch.nn.modules.activation` return the name of the activation."""
    if isinstance(activation, str):
        return activation

    mapper = {
        nn.LeakyReLU: "leaky_relu",
        nn.ReLU: "relu",
        nn.Tanh: "tanh",
        nn.Sigmoid: "sigmoid",
        nn.Softmax: "sigmoid",
    }
    for k, v in mapper.items():
        if isinstance(activation, k):
            return k

    raise ValueError("Unkown given activation type : {}".format(activation))


def get_gain(activation):
    """Given an object of `torch.nn.modules.activation` or an activation name
    return the correct gain."""
    if activation is None:
        return 1

    activation_name = get_activation_name(activation)

    param = None if activation_name != "leaky_relu" else activation.negative_slope
    gain = nn.init.calculate_gain(activation_name, param)

    return gain

def weights_init(module, **kwargs):
    """Initialize a module and all its descendents.

    Parameters
    ----------
    module : nn.Module
       module to initialize.
    """
    module.is_resetted = True
    for m in module.modules():
        try:
            if hasattr(module, "reset_parameters") and module.is_resetted:
                # don't reset if resetted already (might want special)
                continue
        except AttributeError:
            pass

        if isinstance(m, torch.nn.modules.conv._ConvNd):
            # used in https://github.com/brain-research/realistic-ssl-evaluation/
            nn.init.kaiming_normal_(m.weight, mode="fan_out", **kwargs)
        elif isinstance(m, nn.Linear):
            linear_init(m, **kwargs)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def linear_init(module, activation="relu"):
    """Initialize a linear layer.

    Parameters
    ----------
    module : nn.Module
       module to initialize.

    activation : `torch.nn.modules.activation` or str, optional
        Activation that will be used on the `module`.
    """
    x = module.weight

    if module.bias is not None:
        module.bias.data.zero_()

    if activation is None:
        return nn.init.xavier_uniform_(x)

    activation_name = get_activation_name(activation)

    if activation_name == "leaky_relu":
        a = 0 if isinstance(activation, str) else activation.negative_slope
        return nn.init.kaiming_uniform_(x, a=a, nonlinearity="leaky_relu")
    elif activation_name == "relu":
        return nn.init.kaiming_uniform_(x, nonlinearity="relu")
    elif activation_name in ["sigmoid", "tanh"]:
        return nn.init.xavier_uniform_(x, gain=get_gain(activation))


def make_depth_sep_conv(Conv):
    """Make a convolution module depth separable."""

    class DepthSepConv(nn.Module):
        """Make a convolution depth separable.

        Parameters
        ----------
        in_channels : int
            Number of input channels.

        out_channels : int
            Number of output channels.

        kernel_size : int

        **kwargs :
            Additional arguments to `Conv`
        """

        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            confidence=False,
            bias=True,
            **kwargs
        ):
            super().__init__()
            self.depthwise = Conv(
                in_channels,
                in_channels,
                kernel_size,
                groups=in_channels,
                bias=bias,
                **kwargs
            )
            self.pointwise = Conv(in_channels, out_channels, 1, bias=bias)
            self.reset_parameters()

        def forward(self, x):
            out = self.depthwise(x)
            out = self.pointwise(out)
            return out

        def reset_parameters(self):
            weights_init(self)

    return DepthSepConv


def channels_to_2nd_dim(X):
    """
    Takes a signal with channels on the last dimension (for most operations) and
    returns it with channels on the second dimension (for convolutions).
    """
    return X.permute(*([0, X.dim() - 1] + list(range(1, X.dim() - 1))))


def channels_to_last_dim(X):
    """
    Takes a signal with channels on the second dimension (for convolutions) and
    returns it with channels on the last dimension (for most operations).
    """
    return X.permute(*([0] + list(range(2, X.dim())) + [1]))



def grid_to_number_points(X):
    H, W, dim = X.shape[-3:]
    X = X.view(*X.shape[:-3], H * W, dim)
    return X


def collapse_z_samples_batch(t):
    """Merge n_z_samples and batch_size in a single dimension."""
    n_z_samples, batch_size, *rest = t.shape
    return t.contiguous().view(n_z_samples * batch_size, *rest)


def extract_z_samples_batch(t, n_z_samples, batch_size):
    """`reverses` collapse_z_samples_batch."""
    _, *rest = t.shape
    return t.view(n_z_samples, batch_size, *rest)


def replicate_z_samples(t, n_z_samples):
    """Replicates a tensor `n_z_samples` times on a new first dim."""
    return t.unsqueeze(0).expand(n_z_samples, *t.shape)


def pool_and_replicate_middle(t):
    """Mean pools a tensor on all but the first and last dimension (i.e. all the middle dimension)."""
    first, *middle, last = t.shape

    # size = [first, 1, last]
    t = t.view(first, prod(middle), last).mean(1, keepdim=True)

    t = t.view(first, *([1] * len(middle)), last)
    t = t.expand(first, *middle, last)

    # size = [first, *middle, last]
    return t


def prod(iterable):
    """Compute the product of all elements in an iterable."""
    return reduce(operator.mul, iterable, 1)


def make_abs_conv(Conv):
    """Make a convolution have only positive parameters."""

    class AbsConv(Conv):
        def forward(self, input):
            return F.conv2d(
                input,
                self.weight.abs(),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    return AbsConv


def MultivariateNormalDiag(loc, scale_diag):
    """Multi variate Gaussian with a diagonal covariance function (on the last dimension)."""
    if loc.dim() < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return Normal(loc, scale_diag)

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
        decoder_latent_dim=None,
        is_heteroskedastic=True,
        XEncoder=None,
        Decoder=None,
        PredictiveDistribution=Normal,
        p_y_loc_transformer=nn.Identity(),
        p_y_scale_transformer=lambda y_scale: 0.1 + 0.9 * F.softplus(y_scale),
        use_raw=False
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.encoded_path = encoded_path
        self.is_heteroskedastic = is_heteroskedastic
        self.use_raw = use_raw
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
        if decoder_latent_dim is not None:
            self.decoder = Decoder(self.x_transf_dim, decoder_latent_dim, self.y_dim * 2)
        else:
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

    def forward(self, batch, num_samples = None, reduce_ll = True):
        X_cntxt, Y_cntxt, X_trgt, Y_trgt = batch.xc, batch.yc, batch.x, batch.y
        if not self.training:
            Y_trgt = None
            self.n_z_samples_test = num_samples
        else:
            self.n_z_samples_train = num_samples
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
        self._validate_inputs(X_cntxt, Y_cntxt, X_trgt, Y_trgt)

        # size = [batch_size, *n_cntxt, x_transf_dim]
        X_cntxt = self.x_encoder(X_cntxt)
        # size = [batch_size, *n_trgt, x_transf_dim]
        X_trgt_raw = X_trgt.clone()
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
        if self.use_raw:
            p_yCc = self.decode(X_trgt_raw, R_trgt)
        else:
            p_yCc = self.decode(X_trgt, R_trgt)

        if 'mask_xc' in batch.keys(): # ConvGridNP
            # reshape them into (n_z, bs, n_points, dim)
            if q_zCc is not None:
                q_zCc = Normal(grid_to_number_points(q_zCc.loc), grid_to_number_points(q_zCc.scale))
            if q_zCct is not None:
                q_zCct = Normal(grid_to_number_points(q_zCct.loc), grid_to_number_points(q_zCct.scale))
            z_samples = grid_to_number_points(z_samples)
            p_yCc = Normal(grid_to_number_points(p_yCc.loc), grid_to_number_points(p_yCc.scale))
        return p_yCc, z_samples, q_zCc, q_zCct

    def _validate_inputs(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt):
        """Validates the inputs by checking if features are rescaled to [-1,1] during training."""
        if self.training:
            if not (isin_range(X_cntxt, [-2, 2]) and isin_range(X_trgt, [-2, 2])):
                raise ValueError(
                    f"Features during training should be in [-1,1]. {X_cntxt.min()} <= X_cntxt <= {X_cntxt.max()} ; {X_trgt.min()} <= X_trgt <= {X_trgt.max()}."
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
        LatentDistribution=Normal,
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
            self.r_z_merger = nn.Linear(self.r_dim + self.z_dim, self.z_dim)

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


class MLP(nn.Module):
    """General MLP class.

    Parameters
    ----------
    input_size: int

    output_size: int

    hidden_size: int, optional
        Number of hidden neurones.

    n_hidden_layers: int, optional
        Number of hidden layers.

    activation: callable, optional
        Activation function. E.g. `nn.RelU()`.

    is_bias: bool, optional
        Whether to use biaises in the hidden layers.

    dropout: float, optional
        Dropout rate.

    is_force_hid_smaller : bool, optional
        Whether to force the hidden dimensions to be smaller or equal than in and out.
        If not, it forces the hidden dimension to be larger or equal than in or out.

    is_res : bool, optional
        Whether to use residual connections.
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=32,
        n_hidden_layers=1,
        activation=nn.ReLU(),
        is_bias=True,
        dropout=0,
        is_force_hid_smaller=False,
        is_res=False,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.is_res = is_res

        if is_force_hid_smaller and self.hidden_size > max(
            self.output_size, self.input_size
        ):
            self.hidden_size = max(self.output_size, self.input_size)
            txt = "hidden_size={} larger than output={} and input={}. Setting it to {}."
            warnings.warn(
                txt.format(hidden_size, output_size, input_size, self.hidden_size)
            )
        elif self.hidden_size < min(self.output_size, self.input_size):
            self.hidden_size = min(self.output_size, self.input_size)
            txt = (
                "hidden_size={} smaller than output={} and input={}. Setting it to {}."
            )
            warnings.warn(
                txt.format(hidden_size, output_size, input_size, self.hidden_size)
            )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.activation = activation

        self.to_hidden = nn.Linear(self.input_size, self.hidden_size, bias=is_bias)
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size, bias=is_bias)
                for _ in range(self.n_hidden_layers - 1)
            ]
        )
        self.out = nn.Linear(self.hidden_size, self.output_size, bias=is_bias)

        self.reset_parameters()

    def forward(self, x):
        out = self.to_hidden(x)
        out = self.activation(out)
        x = self.dropout(out)

        for linear in self.linears:
            out = linear(x)
            out = self.activation(out)
            if self.is_res:
                out = out + x
            out = self.dropout(out)
            x = out

        out = self.out(x)
        return out

    def reset_parameters(self):
        linear_init(self.to_hidden, activation=self.activation)
        for lin in self.linears:
            linear_init(lin, activation=self.activation)
        linear_init(self.out)


class MergeFlatInputs(nn.Module):
    """
    Extend a module to take 2 flat inputs. It simply returns
    the concatenated flat inputs to the module `module({x1; x2})`.

    Parameters
    ----------
    FlatModule: nn.Module
        Module which takes a non flat inputs.

    x1_dim: int
        Dimensionality of the first flat inputs.

    x2_dim: int
        Dimensionality of the second flat inputs.

    n_out: int
        Size of ouput.

    is_sum_merge : bool, optional
        Whether to transform `flat_input` by an MLP first (if need to resize),
        then sum to `X` (instead of concatenating): useful if the difference in
        dimension between both inputs is very large => don't want one layer to
        depend only on a few dimension of a large input.

    kwargs:
        Additional arguments to FlatModule.
    """

    def __init__(self, FlatModule, x1_dim, x2_dim, n_out, is_sum_merge=False, **kwargs):
        super().__init__()
        self.is_sum_merge = is_sum_merge

        if self.is_sum_merge:
            dim = x1_dim
            self.resizer = MLP(x2_dim, dim)  # transform to be the correct size
        else:
            dim = x1_dim + x2_dim

        self.flat_module = FlatModule(dim, n_out, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, x1, x2):
        if self.is_sum_merge:
            x2 = self.resizer(x2)
            # use activation because if not 2 linear layers in a row => useless computation
            out = torch.relu(x1 + x2)
        else:
            out = torch.cat((x1, x2), dim=-1)

        return self.flat_module(out)


def merge_flat_input(module, is_sum_merge=False, **kwargs):
    """
    Extend a module to accept an additional flat input. I.e. the output should
    be called by `merge_flat_input(module)(x_shape, flat_dim, n_out, **kwargs)`.

    Notes
    -----
    - if x_shape is an integer (currently only available option), it simply returns
    the concatenated flat inputs to the module `module({x; flat_input})`.
    - if `is_sum_merge` then transform `flat_input` by an MLP first, then sum
    to `X` (instead of concatenating): useful if the difference in dimension
    between both inputs is very large => don't want one layer to depend only on
    a few dimension of a large input.
    """

    def merged_flat_input(x_shape, flat_dim, n_out, **kwargs2):
        assert isinstance(x_shape, int)
        return MergeFlatInputs(
            module,
            x_shape,
            flat_dim,
            n_out,
            is_sum_merge=is_sum_merge,
            **kwargs2,
            **kwargs
        )

    return merged_flat_input


class DiscardIthArg(nn.Module):
    """
    Helper module which discard the i^th argument of the constructor and forward,
    before being given to `To`.
    """

    def __init__(self, *args, i=0, To=nn.Identity, **kwargs):
        super().__init__()
        self.i = i
        self.destination = To(*self.filter_args(*args), **kwargs)

    def filter_args(self, *args):
        return [arg for i, arg in enumerate(args) if i != self.i]

    def forward(self, *args, **kwargs):
        return self.destination(*self.filter_args(*args), **kwargs)


def discard_ith_arg(module, i, **kwargs):
    def discarded_arg(*args, **kwargs2):
        return DiscardIthArg(*args, i=i, To=module, **kwargs, **kwargs2)

    return discarded_arg


class ExpRBF(nn.Module):
    """Exponential radial basis function.

    Parameters
    ----------
    x_dim : int
        Dimensionality of input. Placeholder (not used)

    max_dist : float, optional
        Max distance between the closest query and target, used for intialisation.

    max_dist_weight : float, optional
        Weight that should be given to a maximum distance. Note that min_dist_weight
        is 1, so can also be seen as a ratio.

    p : int, optional
        p-norm to use, If p=2, exponential quadratic => Gaussian.
    """

    def __init__(self, x_dim, max_dist=1 / 256, max_dist_weight=0.9, p=2, **kwargs):
        super().__init__()

        self.max_dist = max_dist
        self.max_dist_weight = max_dist_weight
        self.length_scale_param = nn.Parameter(torch.tensor([0.0]))
        self.p = p
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)
        # set the parameter depending on the weight to give to a maxmum distance
        # query. i.e. exp(- (max_dist / sigma).pow(p)) = max_dist_weight
        # => sigma = max_dist / ((- log(max_dist_weight))**(1/p))
        max_dist_sigma = self.max_dist / (
            (-math.log(self.max_dist_weight)) ** (1 / self.p)
        )
        # inverse_softplus : log(exp(y) - 1)
        max_dist_param = math.log(math.exp(max_dist_sigma) - 1)
        self.length_scale_param = nn.Parameter(torch.tensor([max_dist_param]))

    def forward(self, diff):

        # size=[batch_size, n_keys, n_queries, kq_size]
        dist = torch.norm(diff, p=self.p, dim=-1, keepdim=True)

        # compute exponent making sure no division by 0
        sigma = 1e-5 + F.softplus(self.length_scale_param)

        inp = -(dist / sigma).pow(self.p)
        out = torch.softmax(
            inp, dim=-2
        )  # numerically stable normalization of the weights by density

        # size=[batch_size, n_keys, kq_size]
        density = torch.exp(inp).sum(dim=-2)

        return out, density

class SetConv(nn.Module):
    """Applies a convolution over a set of inputs, i.e. generalizes `nn._ConvNd`
    to non uniformly sampled samples [1].

    Parameters
    ----------
    x_dim : int
        Number of spatio-temporal dimensions of input.

    in_channels : int
        Number of input channels.

    out_channels : int
        Number of output channels.

    RadialBasisFunc : callable, optional
        Function which returns the "weight" of each points as a function of their
        distance (i.e. for usual CNN that would be the filter).

    kwargs :
        Additional arguments to `RadialBasisFunc`.

    References
    ----------
    [1] Gordon, Jonathan, et al. "Convolutional conditional neural processes." arXiv preprint
    arXiv:1910.13556 (2019).
    """

    def __init__(
        self, x_dim, in_channels, out_channels, RadialBasisFunc=ExpRBF, **kwargs
    ):
        super().__init__()
        assert x_dim == 1, "Currently only supports single spatial dimension `x_dim==1`"
        self.radial_basis_func = RadialBasisFunc(x_dim, **kwargs)
        self.resizer = nn.Linear(in_channels + 1, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, keys, queries, values):
        """
        Compute the set convolution between {key, value} and {query}.

        TODO
        ----
        - should sort the keys and queries to not compute differences if outside
        of given receptive field (large memory savings).

        Parameters
        ----------
        keys : torch.Tensor, size=[batch_size, n_keys, kq_size]
        queries : torch.Tensor, size=[batch_size, n_queries, kq_size]
        values : torch.Tensor, size=[batch_size, n_keys, in_channels]

        Return
        ------
        targets : torch.Tensor, size=[batch_size, n_queries, out_channels]
        """
        # prepares for broadcasted computations
        keys = keys.unsqueeze(1)
        queries = queries.unsqueeze(2)
        values = values.unsqueeze(1)

        # weight size = [batch_size, n_queries, n_keys, 1]
        # density size = [batch_size, n_queries, 1]
        weight, density = self.radial_basis_func(keys - queries)

        # size = [batch_size, n_queries, value_size]
        targets = (weight * values).sum(dim=2)

        # size = [batch_size, n_queries, value_size+1]
        targets = torch.cat([targets, density], dim=-1)

        return self.resizer(targets)

class CNN(nn.Module):
    """Simple multilayer CNN.

    Parameters
    ----------
    n_channels : int or list
        Number of channels, same for input and output. If list then needs to be
        of size `n_blocks - 1`, e.g. [16, 32, 64] means that you will have a
        `[ConvBlock(16,32), ConvBlock(32, 64)]`.

    ConvBlock : nn.Module
        Convolutional block (unitialized). Needs to take as input `Should be
        initialized with `ConvBlock(in_chan, out_chan)`.

    n_blocks : int, optional
        Number of convolutional blocks.

    is_chan_last : bool, optional
        Whether the channels are on the last dimension of the input.

    kwargs :
        Additional arguments to `ConvBlock`.
    """

    def __init__(self, n_channels, ConvBlock, n_blocks=3, is_chan_last=False, **kwargs):

        super().__init__()
        self.n_blocks = n_blocks
        self.is_chan_last = is_chan_last
        self.in_out_channels = self._get_in_out_channels(n_channels, n_blocks)
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_chan, out_chan, **kwargs)
                for in_chan, out_chan in self.in_out_channels
            ]
        )
        self.is_return_rep = False  # never return representation for vanilla conv

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def _get_in_out_channels(self, n_channels, n_blocks):
        """Return a list of tuple of input and output channels."""
        if isinstance(n_channels, int):
            channel_list = [n_channels] * (n_blocks + 1)
        else:
            channel_list = list(n_channels)

        assert len(channel_list) == (n_blocks + 1), "{} != {}".format(
            len(channel_list), n_blocks + 1
        )

        return list(zip(channel_list, channel_list[1:]))

    def forward(self, X):
        if self.is_chan_last:
            X = channels_to_2nd_dim(X)

        X, representation = self.apply_convs(X)

        if self.is_chan_last:
            X = channels_to_last_dim(X)

        if self.is_return_rep:
            return X, representation

        return X

    def apply_convs(self, X):
        for conv_block in self.conv_blocks:
            X = conv_block(X)
        return X, None


class ResConvBlock(nn.Module):
    """Convolutional block inspired by the pre-activation Resnet [1]
    and depthwise separable convolutions [2].

    Parameters
    ----------
    in_chan : int
        Number of input channels.

    out_chan : int
        Number of output channels.

    Conv : nn.Module
        Convolutional layer (unitialized). E.g. `nn.Conv1d`.

    kernel_size : int or tuple, optional
        Size of the convolving kernel. Should be odd to keep the same size.

    activation: callable, optional
        Activation object. E.g. `nn.RelU()`.

    Normalization : nn.Module, optional
        Normalization layer (unitialized). E.g. `nn.BatchNorm1d`.

    n_conv_layers : int, optional
        Number of convolutional layers, can be 1 or 2.

    is_bias : bool, optional
        Whether to use a bias.

    References
    ----------
    [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016, October). Identity mappings
        in deep residual networks. In European conference on computer vision
        (pp. 630-645). Springer, Cham.

    [2] Chollet, F. (2017). Xception: Deep learning with depthwise separable
        convolutions. In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 1251-1258).
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        Conv,
        kernel_size=5,
        activation=nn.ReLU(),
        Normalization=nn.Identity,
        is_bias=True,
        n_conv_layers=1,
    ):
        super().__init__()
        self.activation = activation
        self.n_conv_layers = n_conv_layers
        assert self.n_conv_layers in [1, 2]

        if kernel_size % 2 == 0:
            raise ValueError("`kernel_size={}`, but should be odd.".format(kernel_size))

        padding = kernel_size // 2

        if self.n_conv_layers == 2:
            self.norm1 = Normalization(in_chan)
            self.conv1 = make_depth_sep_conv(Conv)(
                in_chan, in_chan, kernel_size, padding=padding, bias=is_bias
            )
        self.norm2 = Normalization(in_chan)
        self.conv2_depthwise = Conv(
            in_chan, in_chan, kernel_size, padding=padding, groups=in_chan, bias=is_bias
        )
        self.conv2_pointwise = Conv(in_chan, out_chan, 1, bias=is_bias)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X):

        if self.n_conv_layers == 2:
            out = self.conv1(self.activation(self.norm1(X)))
        else:
            out = X

        out = self.conv2_depthwise(self.activation(self.norm2(out)))
        # adds residual before point wise => output can change number of channels
        out = out + X
        out = self.conv2_pointwise(out.contiguous())  # for some reason need contiguous
        return out