from typing import Callable, Optional, Union

import torch

import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F

from .rotary_embedding import RotaryEmbedding
from .transformer_utils import PositionalEncoding


class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class StyleFeatureEncoder(nn.Module):
    """Lightweight style projection head: Linear -> LayerNorm -> activation."""

    def __init__(
            self,
            dim: int,
            activation: Callable[[Tensor], Tensor] = F.gelu,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.norm(x)
        return self.activation(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
            layer_norm_eps: float = 1e-5,
            batch_first: bool = False,
            norm_first: bool = True,
            device=None,
            dtype=None,
            rotary=None,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
            self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward=2048,
            dropout=0.1,
            activation=F.relu,
            layer_norm_eps=1e-5,
            batch_first=False,
            norm_first=True,
            device=None,
            dtype=None,
            rotary=None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    # x, cond, t
    def forward(
            self,
            tgt,
            memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
    ):
        x = tgt
        if self.norm_first:
            # Preserve the released checkpoint behavior: these branches are
            # evaluated, but the original implementation returned `tgt`.
            self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask)
            )
            x = self.norm2(
                x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
            )
            x = self.norm3(x + self._ff_block(x))
        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    # qkv
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, cond):
        for layer in self.stack:
            x = layer(x, cond)
        return x


# inference keyframe sequence
class MusicDanceCameraKeyframeDecoder(nn.Module):
    def __init__(
            self,
            nfeats: int,
            history_len: int = 60,  # 2 seconds, 30 fps
            inference_len: int = 60,  # 2 seconds, 30 fps
            latent_dim: int = 256,
            ff_size: int = 1024,
            num_layers: int = 4,
            num_heads: int = 4,
            dropout: float = 0.1,
            p_cond_dim: int = 60 * 3,
            m_cond_feature_dim: int = 35,
            vocab_size: int = 2,
            embed_size: int = 2,
            activation: Callable[[Tensor], Tensor] = F.gelu,
            use_rotary=True,
            style_feat_dim: Optional[int] = None,
            **kwargs
    ) -> None:

        super().__init__()

        self.history_len = history_len
        self.inference_len = inference_len
        self.embed_size = embed_size
        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )

        # input projection
        self.keyframe_embed = nn.Embedding(vocab_size, embed_size)
        self.input_projection = nn.Linear(embed_size, latent_dim)

        # conditional projection
        self.p_cond_projection = nn.Linear(p_cond_dim, latent_dim)
        self.m_cond_projection = nn.Linear(m_cond_feature_dim, latent_dim)
        # condition encoder
        self.p_cond_encoder = nn.Sequential()
        self.m_cond_encoder = nn.Sequential()
        # camera history encoder
        self.c_history_encoder = nn.Sequential()
        for _ in range(2):
            self.p_cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
            self.m_cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
            self.c_history_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        self.norm_cond = nn.LayerNorm(latent_dim)
        self.fuse_pm_tokens = nn.Linear(latent_dim * 2, latent_dim)
        self.use_style = style_feat_dim is not None
        if self.use_style:
            self.style_projection = nn.Linear(style_feat_dim, latent_dim)
            self.style_encoder = StyleFeatureEncoder(
                dim=latent_dim,
                activation=activation,
            )
            self.fuse_pms_tokens = nn.Linear(latent_dim * 3, latent_dim)
        else:
            self.style_projection = None
            self.style_encoder = None
            self.fuse_pms_tokens = None
        # decoder
        History2Keyframe_decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            History2Keyframe_decoderstack.append(
                TransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        self.History2KeyframeDecoder = DecoderLayerStack(History2Keyframe_decoderstack)
        self.keyframe_layer = nn.Linear(latent_dim, vocab_size)

    def forward(
            self, x: Tensor, padding_mask: Tensor, p_cond_embed: Tensor, m_cond_embed: Tensor, style_embed: Tensor = None
    ):
        # embed token
        x_embed = self.keyframe_embed(x).reshape(x.shape[0], x.shape[1], self.embed_size)
        # mask inference part
        x_masked = torch.zeros_like(x_embed)
        x_masked[:, :self.history_len, :] = x_embed[:, :self.history_len, :] * padding_mask[:, :self.history_len, :]

        # project to latent space
        x_masked_projection = self.input_projection(x_masked)
        # add the positional embeddings of the input sequence to provide temporal information
        x_masked_projection = self.abs_pos_encoding(x_masked_projection)
        x_masked_tokens = self.c_history_encoder(x_masked_projection)
        x_masked_tokens = self.norm_cond(x_masked_tokens)
        # conditons
        p_cond_tokens = self.p_cond_projection(p_cond_embed)
        p_cond_tokens = self.abs_pos_encoding(p_cond_tokens)
        p_cond_tokens = self.p_cond_encoder(p_cond_tokens)

        m_cond_tokens = self.m_cond_projection(m_cond_embed)
        m_cond_tokens = self.abs_pos_encoding(m_cond_tokens)
        m_cond_tokens = self.m_cond_encoder(m_cond_tokens)

        use_style = (
            self.use_style
            and style_embed is not None
            and style_embed.abs().sum() > 1e-6
        )
        if use_style:
            style_tokens = self.style_projection(style_embed)
            style_tokens = self.abs_pos_encoding(style_tokens)
            style_tokens = self.style_encoder(style_tokens)
            cond_tokens = self.fuse_pms_tokens(torch.cat([p_cond_tokens, m_cond_tokens, style_tokens], dim=-1))
        else:
            cond_tokens = self.fuse_pm_tokens(torch.cat([p_cond_tokens, m_cond_tokens], dim=-1))
        cond_tokens = self.norm_cond(cond_tokens)

        keyframe_seq = self.History2KeyframeDecoder(cond_tokens, x_masked_tokens)
        keyframe_seq = self.keyframe_layer(keyframe_seq)

        return keyframe_seq

class MappingNet(nn.Module):
    """完全借鉴舞蹈生成代码，用于风格条件映射"""

    def __init__(self, in_dim, dim, genre_num):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
        )

        self.unshared = nn.ModuleList()
        for _ in range(genre_num):
            self.unshared.append(nn.Sequential(
                nn.Linear(dim, dim), nn.GELU(),
                nn.Linear(dim, dim)
            ))

    def forward(self, x, genre):
        s = self.shared(x)  # [B, dim]
        s_list = []
        for unshare in self.unshared:
            s_list.append(unshare(s))
        s = torch.stack(s_list, dim=1)  # [B, genre_num, dim]

        genre = genre.squeeze(-1).long()
        batch_indices = torch.arange(len(genre))
        return s[batch_indices, genre]  # [B, dim]

# inference frames and velocity only works for polar representation
class EditableDanceCameraDecoder_ForVelocity(nn.Module):
    def __init__(
            self,
            nfeats: int,
            history_len: int = 60,  # 2 seconds, 30 fps
            inference_len: int = 240,  # 8 seconds, 30 fps
            latent_dim: int = 256,
            ff_size: int = 1024,
            num_layers: int = 4,
            num_heads: int = 4,
            dropout: float = 0.1,
            p_cond_dim: int = 60 * 3,
            m_cond_feature_dim: int = 35,
            activation: Callable[[Tensor], Tensor] = F.gelu,
            use_rotary=True,
            style_feat_dim: Optional[int] = None,
            **kwargs
    ) -> None:

        super().__init__()

        output_feats = nfeats

        self.history_len = history_len
        self.inference_len = inference_len
        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )
        # 添加风格映射网络（可选）
        self.style_mapping = MappingNet(
            in_dim=256,  # 噪声维度
            dim=16,      # 风格嵌入维度
            genre_num=16 # 风格数量
        ) if style_feat_dim is not None else None

        # input projection
        self.input_projection = nn.Linear(nfeats, latent_dim)

        # conditional projection
        self.p_cond_projection = nn.Linear(p_cond_dim, latent_dim)
        self.m_cond_projection = nn.Linear(m_cond_feature_dim, latent_dim)
        # condition encoder
        self.p_cond_encoder = nn.Sequential()
        self.m_cond_encoder = nn.Sequential()
        # camera history encoder
        self.c_history_encoder = nn.Sequential()
        self.use_style = style_feat_dim is not None
        if self.use_style:
            self.style_projection = nn.Linear(style_feat_dim, latent_dim)
            self.style_encoder = StyleFeatureEncoder(
                dim=latent_dim,
                activation=activation,
            )
        else:
            self.style_projection = None
            self.style_encoder = None
        for _ in range(2):
            self.p_cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
            self.m_cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
            self.c_history_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        self.norm_cond = nn.LayerNorm(latent_dim)
        self.fuse_pm_tokens = nn.Linear(latent_dim * 2, latent_dim)
        self.fuse_pms_tokens = nn.Linear(latent_dim * 3, latent_dim) if self.use_style else None  # pose + music + style
        # decoder
        History2Keyframe_decoderstack = nn.ModuleList([])
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            History2Keyframe_decoderstack.append(
                TransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
            decoderstack.append(
                TransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        self.History2KeyframeDecoder = DecoderLayerStack(History2Keyframe_decoderstack)
        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        self.keyframe_layer = nn.Linear(latent_dim, output_feats)
        self.final_layer = nn.Linear(latent_dim, 1)

    def forward(
            self, x: Tensor, x_imask: Tensor, p_cond_embed: Tensor, m_cond_embed: Tensor, style_embed: Optional[Tensor] = None
    ):
        # mask inference part
        x_masked = torch.zeros_like(x)
        x_masked[:, :self.history_len, :] = x[:, :self.history_len, :]

        # project to latent space
        x_masked_projection = self.input_projection(x_masked)
        # add the positional embeddings of the input sequence to provide temporal information
        x_masked_projection = self.abs_pos_encoding(x_masked_projection)
        x_masked_tokens = self.c_history_encoder(x_masked_projection)
        x_masked_tokens = self.norm_cond(x_masked_tokens)
        # conditons
        p_cond_tokens = self.p_cond_projection(p_cond_embed)
        p_cond_tokens = self.abs_pos_encoding(p_cond_tokens)
        p_cond_tokens = self.p_cond_encoder(p_cond_tokens)

        m_cond_tokens = self.m_cond_projection(m_cond_embed)
        m_cond_tokens = self.abs_pos_encoding(m_cond_tokens)
        m_cond_tokens = self.m_cond_encoder(m_cond_tokens)

        use_style = (
            self.use_style
            and style_embed is not None
            and style_embed.abs().sum() > 1e-6
        )
        if use_style:
            style_tokens = self.style_projection(style_embed)  # [B, seq_len, 16] → [B, seq_len, latent_dim]
            style_tokens = self.abs_pos_encoding(style_tokens)  # 加位置编码
            style_tokens = self.style_encoder(style_tokens)  # 编码风格特征
            cond_tokens = self.fuse_pms_tokens(torch.cat([p_cond_tokens, m_cond_tokens, style_tokens], dim=-1))
        else:
            cond_tokens = self.fuse_pm_tokens(torch.cat([p_cond_tokens, m_cond_tokens], dim=-1))
        cond_tokens = self.norm_cond(cond_tokens)

        seq_with_keyframe = self.History2KeyframeDecoder(cond_tokens, x_masked_tokens)
        seq_with_keyframe = self.keyframe_layer(seq_with_keyframe)
        # here we get keyframe, in the following we plan to inference velocity

        # next we use keyframe to synthesize velocity sequence
        x_kmask = torch.zeros_like(x_imask)
        x_kmask[:, self.history_len:-1] = x_imask[:, self.history_len:-1] - x_imask[:, self.history_len + 1:]
        x_kmask[:, self.history_len] += 1.0
        x_kmask[:, -1] = x_imask[:, -1].clone()
        x_masked_with_keyframe = torch.cat([x_masked[:, :self.history_len, :],
                                            seq_with_keyframe[:, self.history_len:, :] * x_kmask[:, self.history_len:,
                                                                                         :]],
                                           dim=1)  # here for inference part, we keep only the start and end keyframe camera motions

        # project to latent space
        x_masked_with_keyframe_projection = self.input_projection(x_masked_with_keyframe)
        # add the positional embeddings of the input sequence to provide temporal information
        x_masked_with_keyframe_projection = self.abs_pos_encoding(x_masked_with_keyframe_projection)
        x_masked_with_keyframe_tokens = self.c_history_encoder(x_masked_with_keyframe_projection)
        x_masked_with_keyframe_tokens = self.norm_cond(x_masked_with_keyframe_tokens)
        x_velocity = self.seqTransDecoder(cond_tokens, x_masked_with_keyframe_tokens)
        x_velocity = self.final_layer(x_velocity)

        # then mask and normalize
        x_imask_v = torch.zeros_like(x_imask)
        x_imask_v[:, self.history_len:-1] = x_imask[:, self.history_len + 1:]
        x_imask_v[:, self.history_len:self.history_len + 1] = x_imask[:, self.history_len:self.history_len + 1]
        x_velocity[:, :self.history_len] *= 0
        x_velocity_min, _ = torch.min(x_velocity, dim=1, keepdim=True)
        x_velocity_non_negative = x_velocity - x_velocity_min + 1e-7

        x_velocity_non_negative_min, _ = torch.min(x_velocity_non_negative, dim=1, keepdim=True)

        x_velocity_masked = x_velocity_non_negative * x_imask_v
        # if torch.any(torch.isnan(x_velocity_masked)):
        #     print("x_velocity_masked")
        x_velocity_normalized = x_velocity_masked / torch.sum(x_velocity_masked, dim=1, keepdim=True)
        # if torch.any(torch.isnan(x_velocity_normalized)):
        #     # print("x_velocity_normalized",torch.sum(x_velocity_masked, dim=1, keepdim=True))
        #     print("x_velocity_normalized", torch.count_nonzero(x_velocity_masked).item(), torch.count_nonzero(x_velocity_non_negative).item())
        x_rho = torch.zeros_like(x_velocity_normalized)

        x_rho[:, self.history_len:] = torch.cumsum(x_velocity_normalized[:, self.history_len:], dim=1)
        output = torch.zeros_like(x_masked_with_keyframe)
        output[:, :self.history_len] += x_masked[:, :self.history_len, :]
        output[:, self.history_len:] += seq_with_keyframe[:, self.history_len:self.history_len + 1, :]
        output[:, self.history_len + 1:] += x_rho[:, self.history_len:-1] * (
                    torch.sum(seq_with_keyframe[:, self.history_len + 1:, :] * x_kmask[:, self.history_len + 1:, :],
                              dim=1, keepdim=True) - seq_with_keyframe[:, self.history_len:self.history_len + 1, :])

        return output

    def forward_with_keyframe(
            self, x: Tensor, x_imask: Tensor, p_cond_embed: Tensor, m_cond_embed: Tensor, style_embed: Optional[Tensor] = None
    ):
        # mask inference part
        x_masked = torch.zeros_like(x)
        x_masked[:, :self.history_len, :] = x[:, :self.history_len, :]

        # project to latent space
        x_masked_projection = self.input_projection(x_masked)
        # add the positional embeddings of the input sequence to provide temporal information
        x_masked_projection = self.abs_pos_encoding(x_masked_projection)
        x_masked_tokens = self.c_history_encoder(x_masked_projection)
        x_masked_tokens = self.norm_cond(x_masked_tokens)
        # conditons
        p_cond_tokens = self.p_cond_projection(p_cond_embed)
        p_cond_tokens = self.abs_pos_encoding(p_cond_tokens)
        p_cond_tokens = self.p_cond_encoder(p_cond_tokens)

        m_cond_tokens = self.m_cond_projection(m_cond_embed)
        m_cond_tokens = self.abs_pos_encoding(m_cond_tokens)
        m_cond_tokens = self.m_cond_encoder(m_cond_tokens)

        use_style = (
            self.use_style
            and style_embed is not None
            and style_embed.abs().sum() > 1e-6
        )
        if use_style:
            style_tokens = self.style_projection(style_embed)
            style_tokens = self.abs_pos_encoding(style_tokens)
            style_tokens = self.style_encoder(style_tokens)
            cond_tokens = self.fuse_pms_tokens(torch.cat([p_cond_tokens, m_cond_tokens, style_tokens], dim=-1))
        else:
            cond_tokens = self.fuse_pm_tokens(torch.cat([p_cond_tokens, m_cond_tokens], dim=-1))
        cond_tokens = self.norm_cond(cond_tokens)

        # seq_with_keyframe = self.History2KeyframeDecoder(cond_tokens, x_masked_tokens)
        # seq_with_keyframe = self.keyframe_layer(seq_with_keyframe)
        seq_with_keyframe = x.clone()
        # here we get keyframe, in the following we plan to inference velocity

        # next we use keyframe to synthesize velocity sequence
        x_kmask = torch.zeros_like(x_imask)
        x_kmask[:, self.history_len:-1] = x_imask[:, self.history_len:-1] - x_imask[:, self.history_len + 1:]
        x_kmask[:, self.history_len] += 1.0
        x_kmask[:, -1] = x_imask[:, -1].clone()
        x_masked_with_keyframe = torch.cat([x_masked[:, :self.history_len, :],
                                            seq_with_keyframe[:, self.history_len:, :] * x_kmask[:, self.history_len:,
                                                                                         :]], dim=1)

        # project to latent space
        x_masked_with_keyframe_projection = self.input_projection(x_masked_with_keyframe)
        # add the positional embeddings of the input sequence to provide temporal information
        x_masked_with_keyframe_projection = self.abs_pos_encoding(x_masked_with_keyframe_projection)
        x_masked_with_keyframe_tokens = self.c_history_encoder(x_masked_with_keyframe_projection)
        x_masked_with_keyframe_tokens = self.norm_cond(x_masked_with_keyframe_tokens)
        x_velocity = self.seqTransDecoder(cond_tokens, x_masked_with_keyframe_tokens)
        # print('output:', output.shape, output[0,:5,:1])
        x_velocity = self.final_layer(x_velocity)

        # then mask and normalize
        # print(x_imask.shape)
        x_imask_v = torch.zeros_like(x_imask)
        x_imask_v[:, self.history_len:-1] = x_imask[:, self.history_len + 1:]
        x_imask_v[:, self.history_len:self.history_len + 1] = x_imask[:, self.history_len:self.history_len + 1]
        x_velocity[:, :self.history_len] *= 0
        x_velocity_min, _ = torch.min(x_velocity, dim=1, keepdim=True)
        x_velocity_non_negative = x_velocity - x_velocity_min + 1e-7
        # debug
        x_velocity_non_negative_min, _ = torch.min(x_velocity_non_negative, dim=1, keepdim=True)

        x_velocity_masked = x_velocity_non_negative * x_imask_v
        # if torch.any(torch.isnan(x_velocity_masked)):
        #     print("x_velocity_masked")
        x_velocity_normalized = x_velocity_masked / torch.sum(x_velocity_masked, dim=1, keepdim=True)
        # if torch.any(torch.isnan(x_velocity_normalized)):
        #     print("x_velocity_normalized", torch.count_nonzero(x_velocity_masked).item(), torch.count_nonzero(x_velocity_non_negative).item())
        x_rho = torch.zeros_like(x_velocity_normalized)

        x_rho[:, self.history_len:] = torch.cumsum(x_velocity_normalized[:, self.history_len:], dim=1)
        output = torch.zeros_like(x_masked_with_keyframe)
        output[:, :self.history_len] += x_masked[:, :self.history_len, :]
        output[:, self.history_len:] += seq_with_keyframe[:, self.history_len:self.history_len + 1, :]
        output[:, self.history_len + 1:] += x_rho[:, self.history_len:-1] * (
                    torch.sum(seq_with_keyframe[:, self.history_len + 1:, :] * x_kmask[:, self.history_len + 1:, :],
                              dim=1, keepdim=True) - seq_with_keyframe[:, self.history_len:self.history_len + 1, :])

        return output
