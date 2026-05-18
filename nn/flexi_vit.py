"""Model code for the OlmoEarth Encoder - inference only version."""

import math

import torch
from einops import rearrange, repeat
from torch import nn

from data.constants import BASE_GSD, Modality, ModalitySpec, get_modality_specs_from_names, MaskValue
from nn.attention import Block
from nn.encodings import (
    get_1d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding_with_resolution,
    get_month_encoding_table,
)
from nn.flexi_patch_embed import FlexiPatchEmbed


def get_modalities_to_process(available_modalities: list[str], supported_modality_names: list[str]) -> list[str]:
    return [mod for mod in supported_modality_names if mod in available_modalities]


class MultiModalPatchEmbeddings(nn.Module):
    def __init__(self, supported_modality_names: list[str], max_patch_size: int, embedding_size: int, use_linear_patch_embed: bool = False):
        super().__init__()
        self.embedding_size = embedding_size
        self.per_modality_embeddings = nn.ModuleDict({})

        for modality in supported_modality_names:
            mod_spec = Modality.get(modality)
            bandset_indices = mod_spec.bandsets_as_indices()
            
            embed_dict = {
                f"{modality}__{idx}": FlexiPatchEmbed(
                    in_chans=len(channel_set_idxs),
                    embedding_size=self.embedding_size,
                    base_patch_size_at_16=max_patch_size,
                    modality_spec=mod_spec,
                    use_linear_patch_embed=use_linear_patch_embed,
                )
                for idx, channel_set_idxs in enumerate(bandset_indices)
            }
            self.per_modality_embeddings[modality] = nn.ModuleDict(embed_dict)
            
            for idx, channel_set_idxs in enumerate(bandset_indices):
                self.register_buffer(f"{modality}__{idx}_buffer", torch.tensor(channel_set_idxs, dtype=torch.long), persistent=False)

    def forward(self, input_data: dict, present_modalities: list[str], patch_size: int) -> dict:
        output_dict = {}
        for modality in present_modalities:
            mod_spec = Modality.get(modality)
            modality_mask = input_data[f"{modality}_mask"]
            modality_data = input_data[modality]
            
            modality_tokens, modality_masks = [], []
            for idx in range(mod_spec.num_band_sets):
                inp_data = torch.index_select(modality_data, -1, getattr(self, f"{modality}__{idx}_buffer"))
                factor = mod_spec.image_tile_size_factor
                token_mask = modality_mask[:, 0::patch_size * factor, 0::patch_size * factor, ..., idx]
                patchified_data = self.per_modality_embeddings[modality][f"{modality}__{idx}"](inp_data, patch_size=patch_size)

                modality_tokens.append(patchified_data)
                modality_masks.append(token_mask)
            
            output_dict[modality] = torch.stack(modality_tokens, dim=-2)
            output_dict[f"{modality}_mask"] = torch.stack(modality_masks, dim=-1)
            
        return output_dict


class CompositeEncodings(nn.Module):
    def __init__(self, embedding_size: int, supported_modalities: list[ModalitySpec], max_timesteps: int):
        super().__init__()
        self.embedding_dim_per_embedding_type = int(embedding_size * 0.25)
        
        self.pos_embed = nn.Parameter(get_1d_sincos_pos_encoding(torch.arange(max_timesteps), self.embedding_dim_per_embedding_type), requires_grad=False)
        self.month_embed = nn.Embedding.from_pretrained(get_month_encoding_table(self.embedding_dim_per_embedding_type), freeze=True)
        
        self.per_modality_channel_embeddings = nn.ParameterDict()
        for modality in supported_modalities:
            shape = (modality.num_band_sets, self.embedding_dim_per_embedding_type)
            self.per_modality_channel_embeddings[modality.name] = nn.Parameter(torch.zeros(shape), requires_grad=False)

    def forward(self, tokens_dict: dict, present_modalities: list[str], timestamps, patch_size: int, input_res: int = BASE_GSD) -> dict:
        output_dict = {}
        for mod_name in present_modalities:
            tokens = tokens_dict[mod_name]
            B, H, W, T, S, D = tokens.shape
            device = tokens.device
            n = self.embedding_dim_per_embedding_type
            
            embed = torch.zeros_like(tokens)
            
            # 1. Modality Embedding
            channel_emb = repeat(self.per_modality_channel_embeddings[mod_name], "s d -> b h w t s d", b=B, h=H, w=W, t=T)
            embed[..., :n] = channel_emb
            
            # 2. Time Embedding
            time_emb = repeat(self.pos_embed[:T], "t d -> b h w t s d", b=B, h=H, w=W, s=S)
            embed[..., n:n*2] = time_emb
            
            # 3. Month Embedding
            month_emb = self.month_embed(timestamps[:, :, 1])
            month_emb = repeat(month_emb, "b t d -> b h w t s d", h=H, w=W, s=S)
            embed[..., n*2:n*3] = month_emb
            
            # 4. Spatial Embedding
            gsd_ratio = input_res * patch_size / BASE_GSD
            sp_emb = get_2d_sincos_pos_encoding_with_resolution((H, W), torch.ones(B, device=device) * gsd_ratio, n, device=device)
            sp_emb = rearrange(sp_emb, "b (h w) d -> b h w d", h=H, w=W)
            sp_emb = repeat(sp_emb, "b h w d -> b h w t s d", t=T, s=S)
            embed[..., n*3:n*4] = sp_emb

            output_dict[mod_name] = tokens + embed
            output_dict[f"{mod_name}_mask"] = tokens_dict[f"{mod_name}_mask"]
            
        return output_dict


class Encoder(nn.Module):
    def __init__(self, embedding_size: int, max_patch_size: int, num_heads: int, mlp_ratio: float, 
                 depth: int, supported_modalities: list, max_timesteps: int, 
                 qk_norm: bool = False, use_linear_patch_embed: bool = False):
        super().__init__()
        self.supported_modality_names = [x.name for x in supported_modalities]
        
        self.patch_embeddings = MultiModalPatchEmbeddings(
            self.supported_modality_names, max_patch_size, embedding_size, use_linear_patch_embed
        )
        self.composite_encodings = CompositeEncodings(
            embedding_size, supported_modalities, max_timesteps
        )
        
        self.blocks = nn.ModuleList([
            Block(
                embedding_size, 
                num_heads, 
                mlp_ratio, 
                qkv_bias=True, 
                qk_norm=qk_norm, 
                norm_layer=nn.LayerNorm,
            ) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embedding_size)

    def forward(self, x: dict, patch_size: int, input_res: int = BASE_GSD) -> dict:
        present_modalities = get_modalities_to_process(list(x.keys()), self.supported_modality_names)
        
        # 1. Patch Embedding & Encoding
        tokens_dict = self.patch_embeddings(x, present_modalities, patch_size)
        encoded_dict = self.composite_encodings(tokens_dict, present_modalities, x["timestamps"], patch_size, input_res)
        
        # 2. Collapse: [B, H, W, T, S, D] -> [B, L, D]
        d_dict = {}
        tokens_list, masks_list = [], []
        for mod in present_modalities:
            d_dict[mod] = encoded_dict[mod].shape
            tokens_list.append(rearrange(encoded_dict[mod], "b h w t s d -> b (h w t s) d"))
            masks_list.append(rearrange(encoded_dict[f"{mod}_mask"], "b h w t s -> b (h w t s)"))
        tokens = torch.cat(tokens_list, dim=1)
        mask = torch.cat(masks_list, dim=1)
        
        attn_mask = (mask == MaskValue.ONLINE_ENCODER.value)
        
        # 3. Transformer Blocks
        for blk in self.blocks:
            tokens = blk(x=tokens, attn_mask=attn_mask)
        
        tokens = self.norm(tokens)
        
        # 4. Split & Expand: [B, L, D] -> [B, H, W, T, S, D]
        out_dict = {}
        start = 0
        for mod in present_modalities:
            # 【修复 1】：只计算空间和时间维度的乘积 (H * W * T * S)，排除最后一个维度 D
            length = math.prod(d_dict[mod][1:-1]) 
            
            # 【修复 2】：在第二维(序列维)切片，并保留第三维(:)，然后 view 成完整的 6D 形状
            out_dict[mod] = tokens[:, start:start+length, :].view(tokens.shape[0], *d_dict[mod][1:])
            out_dict[f"{mod}_mask"] = tokens_dict[f"{mod}_mask"]
            
            start += length
                
        return out_dict
