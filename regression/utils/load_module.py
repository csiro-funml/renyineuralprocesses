import torch
import os
import sys
# Path to directory A
parent_path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(__file__))))
# Add path to A to sys.path
sys.path.append(parent_path)
from models import NP, ANP, TNPA, BANP, TNPD
# from models.convnp import *
from models.new_convnp import *
from models.new_convnp_utils import *

def load_module(filename, config, alpha=0, n_z_samples_test=14, ):
    use_mle = alpha ==0.0
    if filename == 'NP' or filename == "np":
        model = NP(**config, use_mle=use_mle, alpha=alpha)
    if filename == "ANP":
        model = ANP(**config, use_mle=use_mle, alpha=alpha)
    if filename == "TNPD" or filename == 'tnpd':
        model = TNPD(**config, use_mle=use_mle, alpha=alpha)
    if filename == "BANP":
        model = BANP(**config, use_mle=use_mle)
    if filename == "ConvNP":
        if config['dim_x'] == 1:
            R_DIM = 128
            KWARGS = dict(
                is_q_zCct= not use_mle,  # if use_mle, set the posterior sampling to be false
                n_z_samples_train=16,  # going to be more expensive
                n_z_samples_test=n_z_samples_test,
                r_dim=R_DIM,
                Decoder=discard_ith_arg(
                    torch.nn.Linear, i=0
                ),  # use small decoder because already went through CNN
            )

            CNN_KWARGS = dict(
                ConvBlock=ResConvBlock,
                is_chan_last=True,  # all computations are done with channel last in our code
                n_conv_layers=2,
                n_blocks=4,
            )

            # 1D case
            model =ConvNP(
                x_dim=config['dim_x'],
                y_dim=config['dim_y'],
                Interpolator=SetConv,
                CNN=partial(
                    CNN,
                    Conv=torch.nn.Conv1d,
                    Normalization=torch.nn.BatchNorm1d,
                    kernel_size=19,
                    **CNN_KWARGS,
                ),
                density_induced=64,  # density of discretization
                is_global=True,  # use some global representation in addition to local
                **KWARGS,
            )

        else:
            R_DIM = 64
            KWARGS = dict(
                is_q_zCct=not use_mle,
                n_z_samples_train=8,  # going to be more expensive
                n_z_samples_test=16,
                r_dim=R_DIM,
                Decoder=discard_ith_arg(
                    torch.nn.Linear, i=0
                ),  # use small decoder because already went through CNN
            )

            CNN_KWARGS = dict(
                ConvBlock=ResConvBlock,
                is_chan_last=True,  # all computations are done with channel last in our code
                n_conv_layers=2,
                n_blocks=2,
            )

            model = GridConvNP(
                x_dim=1,
                y_dim=config['dim_y'],
                CNN=partial(
                    CNN,
                    Conv=torch.nn.Conv2d,
                    Normalization=torch.nn.BatchNorm2d,
                    kernel_size=9,
                    **CNN_KWARGS,
                ),
                is_global=True,  # use some global representation in addition to local
                **KWARGS,
            )
    return model