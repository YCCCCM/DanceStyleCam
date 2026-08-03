
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # 自注意力
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # MLP
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x


class TransformerEncoder(nn.Module):
    """简化版Transformer编码器 - 减少层数"""

    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MappingNet(nn.Module):
    """简化版MappingNet - 减少复杂度"""

    def __init__(self, in_dim, dim, genre_num):
        super().__init__()
        # 简化共享层
        self.shared = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # 每个风格一个简单的线性映射
        self.unshared = nn.ModuleList()
        for _ in range(genre_num):
            self.unshared.append(nn.Linear(dim, dim))

    def forward(self, x, genre):
        s = self.shared(x)  # [B, dim]
        s_list = []
        for unshare in self.unshared:
            s_list.append(unshare(s))
        s = torch.stack(s_list, dim=1)  # [B, genre_num, dim]

        genre = genre.reshape(-1).long()
        batch_indices = torch.arange(genre.shape[0], device=genre.device)
        return s[batch_indices, genre]  # [B, dim]


class CameraStyleDiscriminator(nn.Module):
    """
    简化版多任务判别器
    保持架构不变，但减少参数
    """

    def __init__(
            self,
            cam_dim: int,  # 摄像机特征维度
            cond_dim: int = 35 + 60 * 3,  # 条件特征维度（音乐35 + 姿势180）
            num_styles: int = 16,  # 风格类别数（16种）
            hidden: int = 128,  # 减少隐藏层维度
    ):
        super().__init__()
        self.num_styles = num_styles

        # 1. 摄像机特征编码器 - 简化
        self.cam_encoder = nn.Sequential(
            spectral_norm(nn.Linear(cam_dim, hidden)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden, hidden)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 2. 条件特征编码器 - 简化
        self.cond_encoder = nn.Sequential(
            spectral_norm(nn.Linear(cond_dim, hidden)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden, hidden)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 3. Transformer时间特征聚合 - 减少层数和宽度
        self.transformer_encoder = TransformerEncoder(
            dim=hidden,
            depth=2,  # 减少深度
            heads=4,  # 减少头数
            mlp_dim=256,  # 减少MLP维度
        )

        # 4. MappingNet（用于真实性判别） - 简化
        self.mapping = MappingNet(
            in_dim=hidden,
            dim=1,
            genre_num=num_styles
        )

        # 5. 风格分类头 - 简化
        self.style_classifier = nn.Sequential(
            spectral_norm(nn.Linear(hidden, hidden // 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden // 2, num_styles),  # 移除spectral_norm以简化
        )

    def forward(self, cam_seq, cond, style_label=None, wrong_style_label=None):
        """
        Args:
            cam_seq: [B, T, cam_dim] 摄像机序列
            cond: [B, T, cond_dim] 条件（音乐+姿势）
            style_label: [B,] 正确风格标签
            wrong_style_label: [B,] 错误风格标签（用于风格聚焦损失）
        Returns:
            如果wrong_style_label不为None:
                (real_logit_correct, real_logit_wrong), (style_logit, None)
            否则:
                real_logit, style_logit
        """
        B, T, _ = cam_seq.shape

        # 编码摄像机特征
        cam_feat = self.cam_encoder(cam_seq)  # [B, T, hidden]

        # 编码条件特征
        cond_feat = self.cond_encoder(cond)  # [B, T, hidden]

        # 融合特征
        x = torch.cat([cam_feat, cond_feat], dim=1)  # [B, 2T, hidden]

        # Transformer编码
        x = self.transformer_encoder(x)  # [B, 2T, hidden]

        # 时间维度平均池化
        x = x.mean(dim=1)  # [B, hidden]

        # 风格分类
        style_logit = self.style_classifier(x)  # [B, num_styles]

        # 处理可能的None情况
        if style_label is None:
            # 如果没有提供风格标签，创建伪标签
            style_label = torch.zeros(B, dtype=torch.long).to(x.device)

        if wrong_style_label is not None:
            # 风格聚焦损失：计算正确和错误风格的真实性分数
            real_logit_correct = self.mapping(x, style_label)  # [B, 1]
            real_logit_wrong = self.mapping(x, wrong_style_label)  # [B, 1]
            return (real_logit_correct, real_logit_wrong), (style_logit, None)
        else:
            # 普通前向
            # real_logit = self.mapping(x, style_label)  # [B, 1]
            real_logit = torch.sigmoid(self.mapping(x, style_label))
            return real_logit, style_logit


# 对抗损失模块保持不变
class CameraAdversarialLoss(nn.Module):
    """对抗损失，包含多种类型选择"""

    def __init__(self, loss_type="wgan"):
        super().__init__()
        self.loss_type = loss_type

        if loss_type == "nsgan":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == "lsgan":
            self.criterion = nn.MSELoss()
        elif loss_type == "hinge":
            self.criterion = nn.ReLU()
        elif loss_type == "wgan":
            self.criterion = None  # WGAN直接使用分数

    def __call__(self, outputs, is_real, is_disc=None):
        if self.loss_type == "hinge":
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        elif self.loss_type == "wgan":
            if is_real:
                return -outputs.mean()
            else:
                return outputs.mean()
        else:
            labels = torch.ones_like(outputs) if is_real else torch.zeros_like(outputs)
            return self.criterion(outputs, labels)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils import spectral_norm
#
#
# class TransformerBlock(nn.Module):
#     def __init__(self, dim, heads, mlp_dim):
#         super().__init__()
#         self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, mlp_dim),
#             nn.GELU(),
#             nn.Linear(mlp_dim, dim)
#         )
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#
#     def forward(self, x):
#         # 自注意力
#         attn_out, _ = self.attention(x, x, x)
#         x = self.norm1(x + attn_out)
#
#         # MLP
#         mlp_out = self.mlp(x)
#         x = self.norm2(x + mlp_out)
#         return x
#
#
# class TransformerEncoder(nn.Module):
#     """简化版Transformer编码器"""
#
#     def __init__(self, dim, depth, heads, mlp_dim):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             TransformerBlock(dim, heads, mlp_dim) for _ in range(depth)
#         ])
#
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x
#
#
# class MappingNet(nn.Module):
#     """完全借鉴舞蹈生成代码，用于风格条件映射"""
#
#     def __init__(self, in_dim, dim, genre_num):
#         super().__init__()
#         self.shared = nn.Sequential(
#             nn.Linear(in_dim, dim), nn.GELU(),
#             nn.Linear(dim, dim), nn.GELU(),
#         )
#
#         self.unshared = nn.ModuleList()
#         for _ in range(genre_num):
#             self.unshared.append(nn.Sequential(
#                 nn.Linear(dim, dim), nn.GELU(),
#                 nn.Linear(dim, dim)
#             ))
#
#     def forward(self, x, genre):
#         s = self.shared(x)  # [B, dim]
#         s_list = []
#         for unshare in self.unshared:
#             s_list.append(unshare(s))
#         s = torch.stack(s_list, dim=1)  # [B, genre_num, dim]
#
#         genre = genre.squeeze(-1).long()
#         batch_indices = torch.arange(len(genre))
#         return s[batch_indices, genre]  # [B, dim]
#
#
# class CameraStyleDiscriminator(nn.Module):
#     """
#     多任务判别器：同时判断真实性和风格分类
#     参考舞蹈生成判别器设计，实现风格聚焦损失
#     """
#
#     def __init__(
#             self,
#             cam_dim: int,  # 摄像机特征维度
#             cond_dim: int = 35 + 60 * 3,  # 条件特征维度（音乐35 + 姿势180）
#             num_styles: int = 16,  # 风格类别数（16种）
#             hidden: int = 256,
#     ):
#         super().__init__()
#         self.num_styles = num_styles
#
#         # 1. 摄像机特征编码器
#         self.cam_encoder = nn.Sequential(
#             spectral_norm(nn.Linear(cam_dim, hidden)),
#             nn.LeakyReLU(0.2, inplace=True),
#             spectral_norm(nn.Linear(hidden, hidden)),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         # 2. 条件特征编码器
#         self.cond_encoder = nn.Sequential(
#             spectral_norm(nn.Linear(cond_dim, hidden)),
#             nn.LeakyReLU(0.2, inplace=True),
#             spectral_norm(nn.Linear(hidden, hidden)),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         # 3. Transformer时间特征聚合
#         self.transformer_encoder = TransformerEncoder(
#             dim=hidden,
#             depth=4,
#             heads=8,
#             mlp_dim=1024,
#         )
#
#         # 4. MappingNet（用于真实性判别）
#         self.mapping = MappingNet(
#             in_dim=hidden,
#             dim=1,
#             genre_num=num_styles
#         )
#
#         # 5. 风格分类头
#         self.style_classifier = nn.Sequential(
#             spectral_norm(nn.Linear(hidden, hidden // 2)),
#             nn.LeakyReLU(0.2, inplace=True),
#             spectral_norm(nn.Linear(hidden // 2, num_styles)),
#         )
#
#     def forward(self, cam_seq, cond, style_label=None, wrong_style_label=None):
#         """
#         Args:
#             cam_seq: [B, T, cam_dim] 摄像机序列
#             cond: [B, T, cond_dim] 条件（音乐+姿势）
#             style_label: [B,] 正确风格标签
#             wrong_style_label: [B,] 错误风格标签（用于风格聚焦损失）
#         Returns:
#             如果wrong_style_label不为None:
#                 (real_logit_correct, real_logit_wrong), (style_logit, None)
#             否则:
#                 real_logit, style_logit
#         """
#         B, T, _ = cam_seq.shape
#
#         # 编码摄像机特征
#         cam_feat = self.cam_encoder(cam_seq)  # [B, T, hidden]
#
#         # 编码条件特征
#         cond_feat = self.cond_encoder(cond)  # [B, T, hidden]
#
#         # 融合特征
#         x = torch.cat([cam_feat, cond_feat], dim=1)  # [B, 2T, hidden]
#
#         # Transformer编码
#         x = self.transformer_encoder(x)  # [B, 2T, hidden]
#
#         # 时间维度平均池化
#         x = x.mean(dim=1)  # [B, hidden]
#
#         # 风格分类
#         style_logit = self.style_classifier(x)  # [B, num_styles]
#
#         # 处理可能的None情况
#         if style_label is None:
#             # 如果没有提供风格标签，创建伪标签
#             style_label = torch.zeros(B, dtype=torch.long).to(x.device)
#             print("Warning: style_label is None, using zeros")
#
#         if wrong_style_label is not None:
#             # 风格聚焦损失：计算正确和错误风格的真实性分数
#             real_logit_correct = self.mapping(x, style_label)  # [B, 1]
#             real_logit_wrong = self.mapping(x, wrong_style_label)  # [B, 1]
#             return (real_logit_correct, real_logit_wrong), (style_logit, None)
#         else:
#             # 普通前向
#             real_logit = self.mapping(x, style_label)  # [B, 1]
#             return real_logit, style_logit
#
#
# class CameraAdversarialLoss(nn.Module):
#     """对抗损失，包含多种类型选择"""
#
#     def __init__(self, loss_type="wgan"):
#         super().__init__()
#         self.loss_type = loss_type
#
#         if loss_type == "nsgan":
#             self.criterion = nn.BCEWithLogitsLoss()
#         elif loss_type == "lsgan":
#             self.criterion = nn.MSELoss()
#         elif loss_type == "hinge":
#             self.criterion = nn.ReLU()
#         elif loss_type == "wgan":
#             self.criterion = None  # WGAN直接使用分数
#
#     def __call__(self, outputs, is_real, is_disc=None):
#         if self.loss_type == "hinge":
#             if is_disc:
#                 if is_real:
#                     outputs = -outputs
#                 return self.criterion(1 + outputs).mean()
#             else:
#                 return (-outputs).mean()
#         elif self.loss_type == "wgan":
#             if is_real:
#                 return -outputs.mean()
#             else:
#                 return outputs.mean()
#         else:
#             labels = torch.ones_like(outputs) if is_real else torch.zeros_like(outputs)
#             return self.criterion(outputs, labels)
