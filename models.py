# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from typing import Dict

from torch import Tensor
from torch.hub import load_state_dict_from_url
import math

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.registry import register_model
from compressai.ops import quantize_ste

# from .denoising_diffusion_pytorch import Unet


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy,
            dtype,
        )

from typing import cast
class CompressionModel(nn.Module):
    """Base class for constructing an auto-encoder with any number of
    EntropyBottleneck or GaussianConditional modules.
    """

    def load_state_dict(self, state_dict, strict=True):
        for name, module in self.named_modules():
            if not any(x.startswith(name) for x in state_dict.keys()):
                continue

            if isinstance(module, EntropyBottleneck):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length"],
                    state_dict,
                )

            if isinstance(module, GaussianConditional):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                    state_dict,
                )

        return nn.Module.load_state_dict(self, state_dict, strict=strict)

    def update(self, scale_table=None, force=False):
        """Updates EntropyBottleneck and GaussianConditional CDFs.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (torch.Tensor): table of scales (i.e. stdev)
                for initializing the Gaussian distributions
                (default: 64 logarithmically spaced scales from 0.11 to 256)
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if at least one of the modules was updated.
        """
        if scale_table is None:
            scale_table = get_scale_table()
        updated = False
        for _, module in self.named_modules():
            if isinstance(module, EntropyBottleneck):
                updated |= module.update(force=force)
            if isinstance(module, GaussianConditional):
                updated |= module.update_scale_table(scale_table, force=force)
        return updated

    def aux_loss(self) -> Tensor:
        r"""Returns the total auxiliary loss over all ``EntropyBottleneck``\s.

        In contrast to the primary "net" loss used by the "net"
        optimizer, the "aux" loss is only used by the "aux" optimizer to
        update *only* the ``EntropyBottleneck.quantiles`` parameters. In
        fact, the "aux" loss does not depend on image data at all.

        The purpose of the "aux" loss is to determine the range within
        which most of the mass of a given distribution is contained, as
        well as its median (i.e. 50% probability). That is, for a given
        distribution, the "aux" loss converges towards satisfying the
        following conditions for some chosen ``tail_mass`` probability:

        * ``cdf(quantiles[0]) = tail_mass / 2``
        * ``cdf(quantiles[1]) = 0.5``
        * ``cdf(quantiles[2]) = 1 - tail_mass / 2``

        This ensures that the concrete ``_quantized_cdf``\s operate
        primarily within a finitely supported region. Any symbols
        outside this range must be coded using some alternative method
        that does *not* involve the ``_quantized_cdf``\s. Luckily, one
        may choose a ``tail_mass`` probability that is sufficiently
        small so that this rarely occurs. It is important that we work
        with ``_quantized_cdf``\s that have a small finite support;
        otherwise, entropy coding runtime performance would suffer.
        Thus, ``tail_mass`` should not be too small, either!
        """
        loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return cast(Tensor, loss)


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

from super_precision import Residual, PreNorm, Attention

class MLPCodec(CompressionModel):
    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(M)

        self.g_a = nn.Sequential(
            conv(3, N),
            nn.ReLU(),
            conv(N, N),
            nn.ReLU(),
            conv(N, N),
            nn.ReLU(),
            conv(N, N),
            Residual(PreNorm(M, Attention(M)))
        )
        # self.to_coef = nn.Linear(M,M)

        self.g_s = nn.Sequential(
            Residual(PreNorm(3*M, Attention(3*M))),
            deconv(M*3, N),
            nn.ReLU(),
            deconv(N, N),
            nn.ReLU(),
            deconv(N, N),
            nn.ReLU(),
            deconv(N, 3),
        )

        # time embeddings
        dim = M
        time_dim = dim * 4

        self.x_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, dim)
        )

        self.y_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, dim)
        )

        self.N = N
        self.M = M

    def forward(self, x):
        y = self.g_a(x)

        B,C,H,W = y.size()

        y_hat, y_likelihoods = self.entropy_bottleneck(y)

        x_coord = torch.arange(H, device = x.device, dtype = torch.long)
        x_emb = self.x_mlp(x_coord).repeat(B,W,1,1).permute(0,3,2,1)

        y_coord = torch.arange(W, device = x.device, dtype = torch.long)
        y_emb = self.y_mlp(y_coord).repeat(B,H,1,1).permute(0,3,1,2)

        latent = torch.cat((y_hat,x_emb,y_emb),1)

        x_hat = self.g_s(latent)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

@register_model("bmshj2018-factorized")
class FactorizedPrior(CompressionModel):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    .. code-block:: none

                  ┌───┐    y
            x ──►─┤g_a├──►─┐
                  └───┘    │
                           ▼
                         ┌─┴─┐
                         │ Q │
                         └─┬─┘
                           │
                     y_hat ▼
                           │
                           ·
                        EB :
                           ·
                           │
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(M)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.N = N
        self.M = M

    @property
    def downsampling_factor(self) -> int:
        return 2**4

    def forward(self, x):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

# @register_model("bmshj2018-factorized")
# class FactorizedPrior(CompressionModel):
#     r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
#     N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
#     <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
#     (ICLR), 2018.

#     .. code-block:: none

#                   ┌───┐    y
#             x ──►─┤g_a├──►─┐
#                   └───┘    │
#                            ▼
#                          ┌─┴─┐
#                          │ Q │
#                          └─┬─┘
#                            │
#                      y_hat ▼
#                            │
#                            ·
#                         EB :
#                            ·
#                            │
#                      y_hat ▼
#                            │
#                   ┌───┐    │
#         x_hat ──◄─┤g_s├────┘
#                   └───┘

#         EB = Entropy bottleneck

#     Args:
#         N (int): Number of channels
#         M (int): Number of channels in the expansion layers (last layer of the
#             encoder and last layer of the hyperprior decoder)
#     """

#     def __init__(self, N, M, **kwargs):
#         super().__init__(**kwargs)

#         self.entropy_bottleneck = EntropyBottleneck(M)
#         self.y_predictor_list = nn.ModuleList([Unet(M, M), Unet(M, M)])

#         self.g_a = nn.Sequential(
#             conv(3, N),
#             GDN(N),
#             conv(N, N),
#             GDN(N),
#             conv(N, N),
#             GDN(N),
#             conv(N, M),
#         )

#         self.g_s = nn.Sequential(
#             deconv(M, N),
#             GDN(N, inverse=True),
#             deconv(N, N),
#             GDN(N, inverse=True),
#             deconv(N, N),
#             GDN(N, inverse=True),
#             deconv(N, 3),
#         )

#         self.N = N
#         self.M = M

#     @property
#     def downsampling_factor(self) -> int:
#         return 2**4

#     def forward(self, x):
#         y = self.g_a(x)
#         y_hat, y_likelihoods = self.entropy_bottleneck(y)
#         y_round = torch.round(y)
#         q_err = y - y_round
#         q_norm = torch.norm(q_err, 2)
#         norm_list = []
#         for i in range(0, 2):
#             b, *_, device = *x.shape, x.device
#             batched_times = torch.full((b,), 0, device = x.device, dtype = torch.long)
#             y_predict = self.y_predictor_list[i](y_round, batched_times) + y_round
#             y_err = y_predict - y.detach()
#             y_norm = torch.norm(y_err, 2)
#             norm_list.append(y_norm)
#             y_round = y_predict.detach()

#         y_hat = y_predict.detach()
#         x_hat = self.g_s(y_hat)

#         return {
#             "x_hat": x_hat,
#             "likelihoods": {
#                 "y": y_likelihoods,
#             },
#             "y_norm1": norm_list[0],
#             "y_norm2": norm_list[1],
#             "norm_sum": sum(norm_list),
#             "q_norm": q_norm,
#         }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        #net.load_state_dict(state_dict)
        net.load_state_dict_whatever(state_dict)
        return net
    
    #Use this if load pretrained model checkpoints
    def load_state_dict_whatever(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name.endswith("._offset") or name.endswith("._quantized_cdf") or name.endswith("._cdf_length") or name.endswith(".scale_table"):
                continue
            if name in own_state and own_state[name].size() == param.size():
                own_state[name].copy_(param)

    def compress(self, x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def optim_parameters(self):
        parameters = []
        parameters += self.g_s.parameters()
        #parameters += self.y_predictor_list.parameters()
        return parameters


@register_model("bmshj2018-factorized-relu")
class FactorizedPriorReLU(FactorizedPrior):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.
    GDN activations are replaced by ReLU.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, 3),
        )


@register_model("bmshj2018-hyperprior")
class ScaleHyperprior(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(N)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)

        #BASELINE
        #net.load_state_dict(state_dict)

        net.load_state_dict_whatever(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

# #TODO: BASELINE

@register_model("mbt2018-mean")
class MeanScaleHyperprior(ScaleHyperprior):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │                 means_hat
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def load_state_dict_whatever(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name.endswith("._offset") or name.endswith("._quantized_cdf") or name.endswith("._cdf_length") or name.endswith(".scale_table"):
                continue
            if name in own_state and own_state[name].size() == param.size():
                own_state[name].copy_(param)


# @register_model("mbt2018-mean")
# class MeanScaleHyperprior(ScaleHyperprior):
#     r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
#     Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
#     Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
#     Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

#     .. code-block:: none

#                   ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
#             x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
#                   └───┘    │     └───┘     └───┘        EB        └───┘ │
#                            ▼                                            │
#                          ┌─┴─┐                                          │
#                          │ Q │                                          ▼
#                          └─┬─┘                                          │
#                            │                                            │
#                      y_hat ▼                                            │
#                            │                                            │
#                            ·                                            │
#                         GC : ◄─────────────────────◄────────────────────┘
#                            ·                 scales_hat
#                            │                 means_hat
#                      y_hat ▼
#                            │
#                   ┌───┐    │
#         x_hat ──◄─┤g_s├────┘
#                   └───┘

#         EB = Entropy bottleneck
#         GC = Gaussian conditional

#     Args:
#         N (int): Number of channels
#         M (int): Number of channels in the expansion layers (last layer of the
#             encoder and last layer of the hyperprior decoder)
#     """

#     def __init__(self, N, M, **kwargs):
#         super().__init__(N=N, M=M, **kwargs)

#         self.upsampler = nn.PixelShuffle(4)
#         self.y_predictor_list = nn.ModuleList([Unet(M, M), Unet(M, M)])

#         self.h_a = nn.Sequential(
#             conv(M, N, stride=1, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             conv(N, N),
#             nn.LeakyReLU(inplace=True),
#             conv(N, N),
#         )

#         self.h_s = nn.Sequential(
#             deconv(N, M),
#             nn.LeakyReLU(inplace=True),
#             deconv(M, M * 3 // 2),
#             nn.LeakyReLU(inplace=True),
#             conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
#         )

#     def forward(self, x):
#         #print("I am using the old model")
#         y = self.g_a(x)
#         z = self.h_a(y)
#         y_round = torch.round(y)
#         q_err = y - y_round
#         q_norm = torch.norm(q_err, 2)
#         norm_list = []
#         for i in range(0, 2):
#             b, *_, device = *x.shape, x.device
#             batched_times = torch.full((b,), 0, device = x.device, dtype = torch.long)
#             y_predict = self.y_predictor_list[i](y_round, batched_times) + y_round
#             y_err = y_predict - y.detach()
#             y_norm = torch.norm(y_err, 2)
#             norm_list.append(y_norm)
#             y_round = y_predict.detach()

#         z_hat, z_likelihoods = self.entropy_bottleneck(z)
#         gaussian_params = self.h_s(z_hat)
#         scales_hat, means_hat = gaussian_params.chunk(2, 1)
#         y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
#         y_hat = y_predict.detach()
#         x_hat = self.g_s(y_hat)

#         return {
#             "x_hat": x_hat,
#             "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
#             "y_norm1": norm_list[0],
#             "y_norm2": norm_list[1],
#             "norm_sum": sum(norm_list),
#             "q_norm": q_norm,
#         }

#     def compress(self, x):
#         y = self.g_a(x)
#         z = self.h_a(y)

#         z_strings = self.entropy_bottleneck.compress(z)
#         z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

#         gaussian_params = self.h_s(z_hat)
#         scales_hat, means_hat = gaussian_params.chunk(2, 1)
#         indexes = self.gaussian_conditional.build_indexes(scales_hat)
#         y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
#         return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

#     def decompress(self, strings, shape):
#         assert isinstance(strings, list) and len(strings) == 2
#         z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
#         gaussian_params = self.h_s(z_hat)
#         scales_hat, means_hat = gaussian_params.chunk(2, 1)
#         indexes = self.gaussian_conditional.build_indexes(scales_hat)
#         y_hat = self.gaussian_conditional.decompress(
#             strings[0], indexes, means=means_hat
#         )
#         x_hat = self.g_s(y_hat).clamp_(0, 1)
#         return {"x_hat": x_hat}

#     def optim_parameters(self):
#         parameters = []
#         parameters += self.g_s.parameters()
#         parameters += self.y_predictor_list.parameters()
#         return parameters

#     #Use this if load pretrained model checkpoints
#     def load_state_dict_whatever(self, state_dict):
#         own_state = self.state_dict()
#         for name, param in state_dict.items():
#             if name.endswith("._offset") or name.endswith("._quantized_cdf") or name.endswith("._cdf_length") or name.endswith(".scale_table"):
#                 continue
#             if name in own_state and own_state[name].size() == param.size():
#                 own_state[name].copy_(param)

# TODO: BASELINE
@register_model("mbt2018")
class JointAutoregressiveHierarchicalPriors(MeanScaleHyperprior):
    r"""Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                   params ▼
                         └─┬─┘                                          │
                     y_hat ▼                  ┌─────┐                   │
                           ├──────────►───────┤  CP ├────────►──────────┤
                           │                  └─────┘                   │
                           ▼                                            ▼
                           │                                            │
                           ·                  ┌─────┐                   │
                        GC : ◄────────◄───────┤  EP ├────────◄──────────┘
                           ·     scales_hat   └─────┘
                           │      means_hat
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional
        EP = Entropy parameters network
        CP = Context prediction (masked convolution)

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

# @register_model("mbt2018")
# class JointAutoregressiveHierarchicalPriors(MeanScaleHyperprior):
#     r"""Joint Autoregressive Hierarchical Priors model from D.
#     Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
#     Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
#     Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

#     .. code-block:: none

#                   ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
#             x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
#                   └───┘    │     └───┘     └───┘        EB        └───┘ │
#                            ▼                                            │
#                          ┌─┴─┐                                          │
#                          │ Q │                                   params ▼
#                          └─┬─┘                                          │
#                      y_hat ▼                  ┌─────┐                   │
#                            ├──────────►───────┤  CP ├────────►──────────┤
#                            │                  └─────┘                   │
#                            ▼                                            ▼
#                            │                                            │
#                            ·                  ┌─────┐                   │
#                         GC : ◄────────◄───────┤  EP ├────────◄──────────┘
#                            ·     scales_hat   └─────┘
#                            │      means_hat
#                      y_hat ▼
#                            │
#                   ┌───┐    │
#         x_hat ──◄─┤g_s├────┘
#                   └───┘

#         EB = Entropy bottleneck
#         GC = Gaussian conditional
#         EP = Entropy parameters network
#         CP = Context prediction (masked convolution)

#     Args:
#         N (int): Number of channels
#         M (int): Number of channels in the expansion layers (last layer of the
#             encoder and last layer of the hyperprior decoder)
#     """

#     def __init__(self, N=192, M=192, **kwargs):
#         super().__init__(N=N, M=M, **kwargs)

#         self.y_predictor_list = nn.ModuleList([Unet(N, N)])

#         self.g_a = nn.Sequential(
#             conv(3, N, kernel_size=5, stride=2),
#             GDN(N),
#             conv(N, N, kernel_size=5, stride=2),
#             GDN(N),
#             conv(N, N, kernel_size=5, stride=2),
#             GDN(N),
#             conv(N, M, kernel_size=5, stride=2),
#         )

#         self.g_s = nn.Sequential(
#             deconv(M, N, kernel_size=5, stride=2),
#             GDN(N, inverse=True),
#             deconv(N, N, kernel_size=5, stride=2),
#             GDN(N, inverse=True),
#             deconv(N, N, kernel_size=5, stride=2),
#             GDN(N, inverse=True),
#             deconv(N, 3, kernel_size=5, stride=2),
#         )

#         self.h_a = nn.Sequential(
#             conv(M, N, stride=1, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             conv(N, N, stride=2, kernel_size=5),
#             nn.LeakyReLU(inplace=True),
#             conv(N, N, stride=2, kernel_size=5),
#         )

#         self.h_s = nn.Sequential(
#             deconv(N, M, stride=2, kernel_size=5),
#             nn.LeakyReLU(inplace=True),
#             deconv(M, M * 3 // 2, stride=2, kernel_size=5),
#             nn.LeakyReLU(inplace=True),
#             conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
#         )

#         self.entropy_parameters = nn.Sequential(
#             nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
#         )

#         self.context_prediction = MaskedConv2d(
#             M, 2 * M, kernel_size=5, padding=2, stride=1
#         )

#         self.gaussian_conditional = GaussianConditional(None)
#         self.N = int(N)
#         self.M = int(M)

#     @property
#     def downsampling_factor(self) -> int:
#         return 2 ** (4 + 2)

#     def forward(self, x):
#         y = self.g_a(x)
#         z = self.h_a(y)
#         z_hat, z_likelihoods = self.entropy_bottleneck(z)
#         y_round = torch.round(y)
#         q_err = y - y_round
#         q_norm = torch.norm(q_err, 2)
#         norm_list = []
#         for i in range(0, 1):
#             b, *_, device = *x.shape, x.device
#             batched_times = torch.full((b,), 0, device = x.device, dtype = torch.long)
#             y_predict = self.y_predictor_list[i](y_round, batched_times) + y_round
#             y_err = y_predict - y.detach()
#             y_norm = torch.norm(y_err, 2)
#             norm_list.append(y_norm)
#             y_round = y_predict.detach()
         
#         params = self.h_s(z_hat)

#         y_hat = self.gaussian_conditional.quantize(
#             y, "noise" if self.training else "dequantize"
#         )
#         ctx_params = self.context_prediction(y_hat)
#         gaussian_params = self.entropy_parameters(
#             torch.cat((params, ctx_params), dim=1)
#         )
#         scales_hat, means_hat = gaussian_params.chunk(2, 1)
#         _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
#         y_hat = y_predict.detach()
#         x_hat = self.g_s(y_hat)

#         return {
#             "x_hat": x_hat,
#             "likelihoods": {
#                 "y": y_likelihoods,
#             },
#             "y_norm1": norm_list[0],
#             #"y_norm2": norm_list[1],
#             "norm_sum": sum(norm_list),
#             "q_norm": q_norm,
#         }

    # # TODO: BASELINE
    # def forward(self, x):
    #     y = self.g_a(x)
    #     z = self.h_a(y)
    #     z_hat, z_likelihoods = self.entropy_bottleneck(z)
    #     params = self.h_s(z_hat)

    #     y_hat = self.gaussian_conditional.quantize(
    #         y, "noise" if self.training else "dequantize"
    #     )
    #     ctx_params = self.context_prediction(y_hat)
    #     gaussian_params = self.entropy_parameters(
    #         torch.cat((params, ctx_params), dim=1)
    #     )
    #     scales_hat, means_hat = gaussian_params.chunk(2, 1)
    #     _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
    #     x_hat = self.g_s(y_hat)

    #     return {
    #         "x_hat": x_hat,
    #         "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
    #     }

#     @classmethod
#     def from_state_dict(cls, state_dict):
#         """Return a new model instance from `state_dict`."""
#         N = state_dict["g_a.0.weight"].size(0)
#         M = state_dict["g_a.6.weight"].size(0)
#         net = cls(N, M)
#         net.load_state_dict_whatever(state_dict)
#         return net

#     #Use this if load pretrained model checkpoints
#     def load_state_dict_whatever(self, state_dict):
#         own_state = self.state_dict()
#         for name, param in state_dict.items():
#             if name.endswith("._offset") or name.endswith("._quantized_cdf") or name.endswith("._cdf_length") or name.endswith(".scale_table"):
#                 continue
#             if name in own_state and own_state[name].size() == param.size():
#                 own_state[name].copy_(param)

#     def optim_parameters(self):
#         parameters = []
#         parameters += self.g_s.parameters()
#         parameters += self.y_predictor_list.parameters()

#         return parameters
# ############################################
#     def compress(self, x):
#         if next(self.parameters()).device != torch.device("cpu"):
#             warnings.warn(
#                 "Inference on GPU is not recommended for the autoregressive "
#                 "models (the entropy coder is run sequentially on CPU).",
#                 stacklevel=2,
#             )

#         y = self.g_a(x)
#         z = self.h_a(y)

#         z_strings = self.entropy_bottleneck.compress(z)
#         z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

#         params = self.h_s(z_hat)

#         s = 4  # scaling factor between z and y
#         kernel_size = 5  # context prediction kernel size
#         padding = (kernel_size - 1) // 2

#         y_height = z_hat.size(2) * s
#         y_width = z_hat.size(3) * s

#         y_hat = F.pad(y, (padding, padding, padding, padding))

#         y_strings = []
#         for i in range(y.size(0)):
#             string = self._compress_ar(
#                 y_hat[i : i + 1],
#                 params[i : i + 1],
#                 y_height,
#                 y_width,
#                 kernel_size,
#                 padding,
#             )
#             y_strings.append(string)

#         return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

#     def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
#         cdf = self.gaussian_conditional.quantized_cdf.tolist()
#         cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
#         offsets = self.gaussian_conditional.offset.tolist()

#         encoder = BufferedRansEncoder()
#         symbols_list = []
#         indexes_list = []

#         # Warning, this is slow...
#         # TODO: profile the calls to the bindings...
#         masked_weight = self.context_prediction.weight * self.context_prediction.mask
#         for h in range(height):
#             for w in range(width):
#                 y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
#                 ctx_p = F.conv2d(
#                     y_crop,
#                     masked_weight,
#                     bias=self.context_prediction.bias,
#                 )

#                 # 1x1 conv for the entropy parameters prediction network, so
#                 # we only keep the elements in the "center"
#                 p = params[:, :, h : h + 1, w : w + 1]
#                 gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
#                 gaussian_params = gaussian_params.squeeze(3).squeeze(2)
#                 scales_hat, means_hat = gaussian_params.chunk(2, 1)

#                 indexes = self.gaussian_conditional.build_indexes(scales_hat)

#                 y_crop = y_crop[:, :, padding, padding]
#                 y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
#                 y_hat[:, :, h + padding, w + padding] = y_q + means_hat

#                 symbols_list.extend(y_q.squeeze().tolist())
#                 indexes_list.extend(indexes.squeeze().tolist())

#         encoder.encode_with_indexes(
#             symbols_list, indexes_list, cdf, cdf_lengths, offsets
#         )

#         string = encoder.flush()
#         return string

#     def decompress(self, strings, shape):
#         assert isinstance(strings, list) and len(strings) == 2

#         if next(self.parameters()).device != torch.device("cpu"):
#             warnings.warn(
#                 "Inference on GPU is not recommended for the autoregressive "
#                 "models (the entropy coder is run sequentially on CPU).",
#                 stacklevel=2,
#             )

#         # FIXME: we don't respect the default entropy coder and directly call the
#         # range ANS decoder

#         z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
#         params = self.h_s(z_hat)

#         s = 4  # scaling factor between z and y
#         kernel_size = 5  # context prediction kernel size
#         padding = (kernel_size - 1) // 2

#         y_height = z_hat.size(2) * s
#         y_width = z_hat.size(3) * s

#         # initialize y_hat to zeros, and pad it so we can directly work with
#         # sub-tensors of size (N, C, kernel size, kernel_size)
#         y_hat = torch.zeros(
#             (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
#             device=z_hat.device,
#         )

#         for i, y_string in enumerate(strings[0]):
#             self._decompress_ar(
#                 y_string,
#                 y_hat[i : i + 1],
#                 params[i : i + 1],
#                 y_height,
#                 y_width,
#                 kernel_size,
#                 padding,
#             )

#         y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
#         x_hat = self.g_s(y_hat).clamp_(0, 1)
#         return {"x_hat": x_hat}

#     def _decompress_ar(
#         self, y_string, y_hat, params, height, width, kernel_size, padding
#     ):
#         cdf = self.gaussian_conditional.quantized_cdf.tolist()
#         cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
#         offsets = self.gaussian_conditional.offset.tolist()

#         decoder = RansDecoder()
#         decoder.set_stream(y_string)

#         # Warning: this is slow due to the auto-regressive nature of the
#         # decoding... See more recent publication where they use an
#         # auto-regressive module on chunks of channels for faster decoding...
#         for h in range(height):
#             for w in range(width):
#                 # only perform the 5x5 convolution on a cropped tensor
#                 # centered in (h, w)
#                 y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
#                 ctx_p = F.conv2d(
#                     y_crop,
#                     self.context_prediction.weight,
#                     bias=self.context_prediction.bias,
#                 )
#                 # 1x1 conv for the entropy parameters prediction network, so
#                 # we only keep the elements in the "center"
#                 p = params[:, :, h : h + 1, w : w + 1]
#                 gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
#                 scales_hat, means_hat = gaussian_params.chunk(2, 1)

#                 indexes = self.gaussian_conditional.build_indexes(scales_hat)
#                 rv = decoder.decode_stream(
#                     indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
#                 )
#                 rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
#                 rv = self.gaussian_conditional.dequantize(rv, means_hat)

#                 hp = h + padding
#                 wp = w + padding
#                 y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv


from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from compressai.registry import register_model


@register_model("cheng2020-anchor")
class Cheng2020Anchor(JointAutoregressiveHierarchicalPriors):
    """Anchor model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses residual blocks with small convolutions (3x3 and 1x1), and sub-pixel
    convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, M=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict_whatever(state_dict)
        return net


@register_model("cheng2020-attn")
class Cheng2020Attention(Cheng2020Anchor):
    """Self-attention model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses self-attention, residual blocks with small convolutions (3x3 and 1x1),
    and sub-pixel convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )


def rename_key(key: str) -> str:
    """Rename state_dict key."""

    # Deal with modules trained with DataParallel
    if key.startswith("module."):
        key = key[7:]

    # ResidualBlockWithStride: 'downsample' -> 'skip'
    if ".downsample." in key:
        return key.replace("downsample", "skip")

    # EntropyBottleneck: nn.ParameterList to nn.Parameters
    if key.startswith("entropy_bottleneck."):
        if key.startswith("entropy_bottleneck._biases."):
            return f"entropy_bottleneck._bias{key[-1]}"

        if key.startswith("entropy_bottleneck._matrices."):
            return f"entropy_bottleneck._matrix{key[-1]}"

        if key.startswith("entropy_bottleneck._factors."):
            return f"entropy_bottleneck._factor{key[-1]}"

    return key


def load_pretrained(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Convert state_dict keys."""
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    return state_dict

__all__ = [
    "bmshj2018_factorized",
    "bmshj2018_factorized_relu",
    "bmshj2018_hyperprior",
    "mbt2018",
    "mbt2018_mean",
    "cheng2020_anchor",
    "cheng2020_attn",
]

model_architectures = {
    "bmshj2018-factorized": FactorizedPrior,
    "bmshj2018_factorized_relu": FactorizedPriorReLU,
    "bmshj2018-hyperprior": ScaleHyperprior,
    "mbt2018-mean": MeanScaleHyperprior,
    "mbt2018": JointAutoregressiveHierarchicalPriors,
    "cheng2020-anchor": Cheng2020Anchor,
    "cheng2020-attn": Cheng2020Attention,
}

root_url = "https://compressai.s3.amazonaws.com/models/v1"
model_urls = {
    "bmshj2018-factorized": {
        "mse": {
            1: f"{root_url}/bmshj2018-factorized-prior-1-446d5c7f.pth.tar",
            2: f"{root_url}/bmshj2018-factorized-prior-2-87279a02.pth.tar",
            3: f"{root_url}/bmshj2018-factorized-prior-3-5c6f152b.pth.tar",
            4: f"{root_url}/bmshj2018-factorized-prior-4-1ed4405a.pth.tar",
            5: f"{root_url}/bmshj2018-factorized-prior-5-866ba797.pth.tar",
            6: f"{root_url}/bmshj2018-factorized-prior-6-9b02ea3a.pth.tar",
            7: f"{root_url}/bmshj2018-factorized-prior-7-6dfd6734.pth.tar",
            8: f"{root_url}/bmshj2018-factorized-prior-8-5232faa3.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/bmshj2018-factorized-ms-ssim-1-9781d705.pth.tar",
            2: f"{root_url}/bmshj2018-factorized-ms-ssim-2-4a584386.pth.tar",
            3: f"{root_url}/bmshj2018-factorized-ms-ssim-3-5352f123.pth.tar",
            4: f"{root_url}/bmshj2018-factorized-ms-ssim-4-4f91b847.pth.tar",
            5: f"{root_url}/bmshj2018-factorized-ms-ssim-5-b3a88897.pth.tar",
            6: f"{root_url}/bmshj2018-factorized-ms-ssim-6-ee028763.pth.tar",
            7: f"{root_url}/bmshj2018-factorized-ms-ssim-7-8c265a29.pth.tar",
            8: f"{root_url}/bmshj2018-factorized-ms-ssim-8-8811bd14.pth.tar",
        },
    },
    "bmshj2018-hyperprior": {
        "mse": {
            1: f"{root_url}/bmshj2018-hyperprior-1-7eb97409.pth.tar",
            2: f"{root_url}/bmshj2018-hyperprior-2-93677231.pth.tar",
            3: f"{root_url}/bmshj2018-hyperprior-3-6d87be32.pth.tar",
            4: f"{root_url}/bmshj2018-hyperprior-4-de1b779c.pth.tar",
            5: f"{root_url}/bmshj2018-hyperprior-5-f8b614e1.pth.tar",
            6: f"{root_url}/bmshj2018-hyperprior-6-1ab9c41e.pth.tar",
            7: f"{root_url}/bmshj2018-hyperprior-7-3804dcbd.pth.tar",
            8: f"{root_url}/bmshj2018-hyperprior-8-a583f0cf.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/bmshj2018-hyperprior-ms-ssim-1-5cf249be.pth.tar",
            2: f"{root_url}/bmshj2018-hyperprior-ms-ssim-2-1ff60d1f.pth.tar",
            3: f"{root_url}/bmshj2018-hyperprior-ms-ssim-3-92dd7878.pth.tar",
            4: f"{root_url}/bmshj2018-hyperprior-ms-ssim-4-4377354e.pth.tar",
            5: f"{root_url}/bmshj2018-hyperprior-ms-ssim-5-c34afc8d.pth.tar",
            6: f"{root_url}/bmshj2018-hyperprior-ms-ssim-6-3a6d8229.pth.tar",
            7: f"{root_url}/bmshj2018-hyperprior-ms-ssim-7-8747d3bc.pth.tar",
            8: f"{root_url}/bmshj2018-hyperprior-ms-ssim-8-cc15b5f3.pth.tar",
        },
    },
    "mbt2018-mean": {
        "mse": {
            1: f"{root_url}/mbt2018-mean-1-e522738d.pth.tar",
            2: f"{root_url}/mbt2018-mean-2-e54a039d.pth.tar",
            3: f"{root_url}/mbt2018-mean-3-723404a8.pth.tar",
            4: f"{root_url}/mbt2018-mean-4-6dba02a3.pth.tar",
            5: f"{root_url}/mbt2018-mean-5-d504e8eb.pth.tar",
            6: f"{root_url}/mbt2018-mean-6-a19628ab.pth.tar",
            7: f"{root_url}/mbt2018-mean-7-d5d441d1.pth.tar",
            8: f"{root_url}/mbt2018-mean-8-8089ae3e.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/mbt2018-mean-ms-ssim-1-5bf9c0b6.pth.tar",
            2: f"{root_url}/mbt2018-mean-ms-ssim-2-e2a1bf3f.pth.tar",
            3: f"{root_url}/mbt2018-mean-ms-ssim-3-640ce819.pth.tar",
            4: f"{root_url}/mbt2018-mean-ms-ssim-4-12626c13.pth.tar",
            5: f"{root_url}/mbt2018-mean-ms-ssim-5-1be7f059.pth.tar",
            6: f"{root_url}/mbt2018-mean-ms-ssim-6-b83bf379.pth.tar",
            7: f"{root_url}/mbt2018-mean-ms-ssim-7-ddf9644c.pth.tar",
            8: f"{root_url}/mbt2018-mean-ms-ssim-8-0cc7b94f.pth.tar",
        },
    },
    "mbt2018": {
        "mse": {
            1: f"{root_url}/mbt2018-1-3f36cd77.pth.tar",
            2: f"{root_url}/mbt2018-2-43b70cdd.pth.tar",
            3: f"{root_url}/mbt2018-3-22901978.pth.tar",
            4: f"{root_url}/mbt2018-4-456e2af9.pth.tar",
            5: f"{root_url}/mbt2018-5-b4a046dd.pth.tar",
            6: f"{root_url}/mbt2018-6-7052e5ea.pth.tar",
            7: f"{root_url}/mbt2018-7-8ba2bf82.pth.tar",
            8: f"{root_url}/mbt2018-8-dd0097aa.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/mbt2018-ms-ssim-1-2878436b.pth.tar",
            2: f"{root_url}/mbt2018-ms-ssim-2-c41cb208.pth.tar",
            3: f"{root_url}/mbt2018-ms-ssim-3-d0dd64e8.pth.tar",
            4: f"{root_url}/mbt2018-ms-ssim-4-a120e037.pth.tar",
            5: f"{root_url}/mbt2018-ms-ssim-5-9b30e3b7.pth.tar",
            6: f"{root_url}/mbt2018-ms-ssim-6-f8b3626f.pth.tar",
            7: f"{root_url}/mbt2018-ms-ssim-7-16e6ff50.pth.tar",
            8: f"{root_url}/mbt2018-ms-ssim-8-0cb49d43.pth.tar",
        },
    },
    "cheng2020-anchor": {
        "mse": {
            1: f"{root_url}/cheng2020-anchor-1-dad2ebff.pth.tar",
            2: f"{root_url}/cheng2020-anchor-2-a29008eb.pth.tar",
            3: f"{root_url}/cheng2020-anchor-3-e49be189.pth.tar",
            4: f"{root_url}/cheng2020-anchor-4-98b0b468.pth.tar",
            5: f"{root_url}/cheng2020-anchor-5-23852949.pth.tar",
            6: f"{root_url}/cheng2020-anchor-6-4c052b1a.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/cheng2020_anchor-ms-ssim-1-20f521db.pth.tar",
            2: f"{root_url}/cheng2020_anchor-ms-ssim-2-c7ff5812.pth.tar",
            3: f"{root_url}/cheng2020_anchor-ms-ssim-3-c23e22d5.pth.tar",
            4: f"{root_url}/cheng2020_anchor-ms-ssim-4-0e658304.pth.tar",
            5: f"{root_url}/cheng2020_anchor-ms-ssim-5-c0a95e77.pth.tar",
            6: f"{root_url}/cheng2020_anchor-ms-ssim-6-f2dc1913.pth.tar",
        },
    },
    "cheng2020-attn": {
        "mse": {
            1: f"{root_url}/cheng2020_attn-mse-1-465f2b64.pth.tar",
            2: f"{root_url}/cheng2020_attn-mse-2-e0805385.pth.tar",
            3: f"{root_url}/cheng2020_attn-mse-3-2d07bbdf.pth.tar",
            4: f"{root_url}/cheng2020_attn-mse-4-f7b0ccf2.pth.tar",
            5: f"{root_url}/cheng2020_attn-mse-5-26c8920e.pth.tar",
            6: f"{root_url}/cheng2020_attn-mse-6-730501f2.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/cheng2020_attn-ms-ssim-1-c5381d91.pth.tar",
            2: f"{root_url}/cheng2020_attn-ms-ssim-2-5dad201d.pth.tar",
            3: f"{root_url}/cheng2020_attn-ms-ssim-3-5c9be841.pth.tar",
            4: f"{root_url}/cheng2020_attn-ms-ssim-4-8b2f647e.pth.tar",
            5: f"{root_url}/cheng2020_attn-ms-ssim-5-5ca1f34c.pth.tar",
            6: f"{root_url}/cheng2020_attn-ms-ssim-6-216423ec.pth.tar",
        },
    },
}

cfgs = {
    "bmshj2018-factorized": {
        1: (128, 192),
        2: (128, 192),
        3: (128, 192),
        4: (128, 192),
        5: (128, 192),
        6: (192, 320),
        7: (192, 320),
        8: (192, 320),
    },
    "bmshj2018-factorized-relu": {
        1: (128, 192),
        2: (128, 192),
        3: (128, 192),
        4: (128, 192),
        5: (128, 192),
        6: (192, 320),
        7: (192, 320),
        8: (192, 320),
    },
    "bmshj2018-hyperprior": {
        1: (128, 192),
        2: (128, 192),
        3: (128, 192),
        4: (128, 192),
        5: (128, 192),
        6: (192, 320),
        7: (192, 320),
        8: (192, 320),
    },
    "mbt2018-mean": {
        1: (128, 192),
        2: (128, 192),
        3: (128, 192),
        4: (128, 192),
        5: (192, 320),
        6: (192, 320),
        7: (192, 320),
        8: (192, 320),
    },
    "mbt2018": {
        1: (192, 192),
        2: (192, 192),
        3: (192, 192),
        4: (192, 192),
        5: (192, 320),
        6: (192, 320),
        7: (192, 320),
        8: (192, 320),
    },
    "cheng2020-anchor": {
        1: (128,),
        2: (128,),
        3: (128,),
        4: (192,),
        5: (192,),
        6: (192,),
    },
    "cheng2020-attn": {
        1: (128,),
        2: (128,),
        3: (128,),
        4: (192,),
        5: (192,),
        6: (192,),
    },
}


def _load_model(
    architecture, metric, quality, pretrained=False, progress=True, **kwargs
):
    if architecture not in model_architectures:
        raise ValueError(f'Invalid architecture name "{architecture}"')

    if quality not in cfgs[architecture]:
        raise ValueError(f'Invalid quality value "{quality}"')

    if pretrained:
        if (
            architecture not in model_urls
            or metric not in model_urls[architecture]
            or quality not in model_urls[architecture][metric]
        ):
            raise RuntimeError("Pre-trained model not yet available")

        url = model_urls[architecture][metric][quality]
        state_dict = load_state_dict_from_url(url, progress=progress)
        state_dict = load_pretrained(state_dict)
        model = model_architectures[architecture].from_state_dict(state_dict)
        return model

    model = model_architectures[architecture](*cfgs[architecture][quality], **kwargs)
    return model


def bmshj2018_factorized(
    quality, metric="mse", pretrained=False, progress=True, **kwargs
):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model(
        "bmshj2018-factorized", metric, quality, pretrained, progress, **kwargs
    )


def bmshj2018_factorized_relu(
    quality, metric="mse", pretrained=False, progress=True, **kwargs
):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.
    GDN activations are replaced by ReLU
    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model(
        "bmshj2018-factorized", metric, quality, pretrained, progress, **kwargs
    )


def bmshj2018_hyperprior(
    quality, metric="mse", pretrained=False, progress=True, **kwargs
):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model(
        "bmshj2018-hyperprior", metric, quality, pretrained, progress, **kwargs
    )


def mbt2018_mean(quality, metric="mse", pretrained=False, progress=True, **kwargs):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model("mbt2018-mean", metric, quality, pretrained, progress, **kwargs)


def mbt2018(quality, metric="mse", pretrained=False, progress=True, **kwargs):
    r"""Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model("mbt2018", metric, quality, pretrained, progress, **kwargs)


def cheng2020_anchor(quality, metric="mse", pretrained=False, progress=True, **kwargs):
    r"""Anchor model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        quality (int): Quality levels (1: lowest, highest: 6)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 6:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 6)')

    return _load_model(
        "cheng2020-anchor", metric, quality, pretrained, progress, **kwargs
    )


def cheng2020_attn(quality, metric="mse", pretrained=False, progress=True, **kwargs):
    r"""Self-attention model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        quality (int): Quality levels (1: lowest, highest: 6)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 6:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 6)')

    return _load_model(
        "cheng2020-attn", metric, quality, pretrained, progress, **kwargs
    )



image_models = {
    "bmshj2018-factorized": bmshj2018_factorized,
    "bmshj2018-factorized-relu": bmshj2018_factorized_relu,
    "bmshj2018-hyperprior": bmshj2018_hyperprior,
    "mbt2018-mean": mbt2018_mean,
    "mbt2018": mbt2018,
    "cheng2020-anchor": cheng2020_anchor,
    "cheng2020-attn": cheng2020_attn,
}