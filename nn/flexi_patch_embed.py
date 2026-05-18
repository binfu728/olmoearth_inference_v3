"""Flexible patch embedding Module.

Extended from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L24
by https://github.com/bwconrad/flexivit/
"""

from collections.abc import Iterable

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from data.constants import ModalitySpec


def _to_2tuple(x: int | tuple[int, ...]) -> tuple[int, int]:
    """Convert a scalar or 2-element iterable to a (h, w) tuple."""
    if isinstance(x, int):
        return (x, x)
    if isinstance(x, Iterable) and not isinstance(x, str):
        values = tuple(x)
        assert len(values) == 2, "x must be a 2-tuple"
        return (int(values[0]), int(values[1]))
    raise TypeError(f"Expected int or tuple[int, int], got {type(x)}")


class FlexiPatchEmbed(nn.Module):
    """Flexible patch embedding nn.Module."""

    def __init__(
        self,
        modality_spec: ModalitySpec,
        base_patch_size_at_16: int | tuple[int, int],
        in_chans: int = 3,
        embedding_size: int = 128,
        norm_layer: nn.Module | None = None,
        bias: bool = True,
        interpolation: str = "bicubic",
        antialias: bool = True,
        use_linear_patch_embed: bool = True,
    ) -> None:
        """2D image to patch embedding w/ flexible patch sizes.

        Extended from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L24
        by https://github.com/bwconrad/flexivit/

        Args:
            modality_spec: The modality spec for this modality
            base_patch_size_at_16: Base patch size. i.e the size of the parameter buffer at a resolution of 16
            in_chans: Number of input image channels
            embedding_size: Network embedding dimension size
            norm_layer: Optional normalization layer
            bias: Whether to use bias in convolution
            interpolation: Resize interpolation type
            antialias: Whether to apply antialiasing resizing
            use_linear_patch_embed: If True, use nn.Linear (reshape + matmul via cuBLAS GEMM).
                If False, use nn.Conv2d (required to load checkpoints trained before this flag existed).
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.use_linear_patch_embed = use_linear_patch_embed

        self.modality_spec = modality_spec
        self.base_patch_size = _to_2tuple(
            base_patch_size_at_16 * modality_spec.image_tile_size_factor
        )

        p_h, p_w = self.base_patch_size
        if use_linear_patch_embed:
            self.proj = nn.Linear(in_chans * p_h * p_w, embedding_size, bias=bias)
            self.proj._skip_custom_init = True
        else:
            self.proj = nn.Conv2d(
                in_chans,
                embedding_size,
                kernel_size=self.base_patch_size,
                stride=self.base_patch_size,
                bias=bias,
            )
        self.norm = norm_layer(embedding_size) if norm_layer else nn.Identity()
        self.interpolation = interpolation
        self.antialias = antialias

    def _resolve_patch_size(
        self, patch_size: int | tuple[int, int] | None
    ) -> tuple[int, int]:
        """Resolve the effective patch size, applying the modality tile size factor."""
        if not patch_size:
            return self.base_patch_size
        if isinstance(patch_size, tuple):
            patch_size = (
                patch_size[0] * self.modality_spec.image_tile_size_factor,
                patch_size[1] * self.modality_spec.image_tile_size_factor,
            )
        else:
            patch_size = patch_size * self.modality_spec.image_tile_size_factor
        resolved = _to_2tuple(patch_size)
        assert isinstance(resolved, tuple) and len(resolved) == 2
        return resolved

    def _project_linear(
        self,
        x: Tensor,
        h_patches: int,
        w_patches: int,
        batch_size: int,
        has_time_dim: bool,
        num_timesteps: int,
    ) -> Tensor:
        """Project patches using nn.Linear (reshape → cuBLAS GEMM → reshape)."""
        p_h, p_w = self.base_patch_size
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p_h, p2=p_w)
        x = self.proj(x)
        if has_time_dim:
            return rearrange(
                x,
                "(b t) (h w) d -> b h w t d",
                b=batch_size,
                t=num_timesteps,
                h=h_patches,
                w=w_patches,
            )
        return rearrange(x, "b (h w) d -> b h w d", h=h_patches, w=w_patches)

    def _project_conv(
        self,
        x: Tensor,
        batch_size: int,
        has_time_dim: bool,
        num_timesteps: int,
    ) -> Tensor:
        """Project patches using nn.Conv2d (for loading pre-linear checkpoints)."""
        x = self.proj(x)
        if has_time_dim:
            _, d, h, w = x.shape
            return rearrange(
                x,
                "(b t) d h w -> b h w t d",
                b=batch_size,
                t=num_timesteps,
                h=h,
                w=w,
            )
        return rearrange(x, "b d h w -> b h w d")

    def forward(
        self,
        x: Tensor,
        patch_size: int | tuple[int, int] | None = None,
    ) -> Tensor:
        """Forward pass for the FlexiPatchEmbed module.

        Args:
            x: Input tensor with shape [b, h, w, (t), c]
            patch_size: Requested patch size to use for the embedding. If None, uses the base patch size.
        """
        batch_size = x.shape[0]
        has_time_dim = len(x.shape) == 5
        num_timesteps = x.shape[3] if has_time_dim else 0

        if has_time_dim:
            x = rearrange(x, "b h w t c -> (b t) c h w")
        else:
            x = rearrange(x, "b h w c -> b c h w")

        req_patch_size = self._resolve_patch_size(patch_size)

        if req_patch_size != self.base_patch_size:
            shape = x.shape[-2:]
            new_shape = (
                shape[0] // req_patch_size[0] * self.base_patch_size[0],
                shape[1] // req_patch_size[1] * self.base_patch_size[1],
            )
            x = F.interpolate(
                x, size=new_shape, mode=self.interpolation, antialias=self.antialias
            )

        p_h, p_w = self.base_patch_size
        h_patches, w_patches = x.shape[-2] // p_h, x.shape[-1] // p_w

        if self.use_linear_patch_embed:
            x = self._project_linear(
                x, h_patches, w_patches, batch_size, has_time_dim, num_timesteps
            )
        else:
            x = self._project_conv(x, batch_size, has_time_dim, num_timesteps)

        return self.norm(x)