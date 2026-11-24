import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
from monai.networks.blocks import UnetOutBlock, UnetrUpBlock
import os


# =========================================================================
# 1. 基础模块 (IDWT, DWT, UpBlock) - 保持不变
# =========================================================================
class IDWT3D(nn.Module):
    def __init__(self, upscale_factor=2):
        super(IDWT3D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        b, c_total, d, h, w = x.shape
        r = self.upscale_factor
        c_out = c_total // (r ** 3)
        x = x.view(b, c_out, r, r, r, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
        x = x.reshape(b, c_out, d * r, h * r, w * r)
        return x


class DWTDownSample3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt_sim = nn.Conv3d(
            in_channels,
            in_channels * 8,
            kernel_size=2, stride=2, groups=in_channels, bias=False
        )
        self.channel_compress = nn.Sequential(
            nn.Conv3d(in_channels * 8, out_channels, kernel_size=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        dwt_output = self.dwt_sim(x)
        out = self.channel_compress(dwt_output)
        return out


class WaveletUnetrUpBlock(nn.Module):
    def __init__(
            self, spatial_dims, in_channels, out_channels,
            kernel_size=3, upsample_kernel_size=2, norm_name="instance", res_block=True
    ):
        super().__init__()
        self.wavelet_generator = nn.Conv3d(in_channels, out_channels * 8, kernel_size=1, bias=False)
        self.idwt = IDWT3D(upscale_factor=2)
        self.conv_block = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=norm_name,
            res_block=res_block,
        ).conv_block

    def forward(self, inp, skip):
        subbands = self.wavelet_generator(inp)
        upsampled = self.idwt(subbands)
        if upsampled.shape[2:] != skip.shape[2:]:
            upsampled = F.interpolate(upsampled, size=skip.shape[2:], mode='nearest')
        concat = torch.cat((upsampled, skip), dim=1)
        out = self.conv_block(concat)
        return out


class SimpleCNNEncoder3D(nn.Module):
    def __init__(self, in_channels=1, base_filters=24):
        super().__init__()
        self.down0 = DWTDownSample3d(in_channels, base_filters)
        self.down1 = DWTDownSample3d(base_filters, base_filters * 2)
        self.down2 = DWTDownSample3d(base_filters * 2, base_filters * 4)
        self.down3 = DWTDownSample3d(base_filters * 4, base_filters * 8)

    def forward(self, x):
        c0 = self.down0(x)
        c1 = self.down1(c0)
        c2 = self.down2(c1)
        c3 = self.down3(c2)
        return [c0, c1, c2, c3]


# =========================================================================
# 🔥 2. [核心创新] Parallel Local-Enhanced MLP (激进增强版)
# =========================================================================
class ParallelLocalMLP(nn.Module):
    """
    🔥 激进版创新点：
    1. 弃用 Depth-wise，改用标准 Conv3d (groups=1)。
       原因：浅层通道数少 (96/192)，显存允许；标准卷积特征提取能力远强于 DW 卷积。
    2. Alpha 初始化改为 0.5。
       原因：强制模型在训练初期就“正视”这个分支，而不是把它当噪点忽略。
    """

    def __init__(self, original_mlp, dim):
        super().__init__()
        self.original_mlp = original_mlp

        # 🔥 升级策略：全通道交互卷积 (Standard Conv)
        self.local_branch = nn.Sequential(
            # 1. 空间建模：Standard Conv3d (groups=1) -> 捕捉最强的局部关联
            # 注意：groups=1 代表全连接卷积，信息交互能力最强
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=1, bias=False),
            nn.InstanceNorm3d(dim),
            nn.GELU(),

            # 2. 通道混合：继续增强非线性
            nn.Conv3d(dim, dim, kernel_size=1, bias=False)
        )

        # 🔥 激进策略：初始化为 0.5
        # 让梯度在第一轮就猛烈地流向卷积分支，避免 "主路霸权"
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # 1. 主路计算 (原版 Swin MLP)
        original_out = self.original_mlp(x)

        # 2. 支路计算 (增强版 3D 卷积)
        local_out = None
        try:
            # (B, D, H, W, C) or (B, C, D, H, W)
            if x.ndim == 5:
                # 判断 channel 位置
                if x.shape[1] == self.local_branch[0].in_channels:
                    # (B, C, D, H, W) 直接卷
                    local_out = self.local_branch(x)
                else:
                    # (B, D, H, W, C) -> Permute -> Conv -> Permute
                    x_img = x.permute(0, 4, 1, 2, 3)
                    local_out = self.local_branch(x_img)
                    local_out = local_out.permute(0, 2, 3, 4, 1)

            # (B, N, C)
            elif x.ndim == 3:
                B, N, C = x.shape
                # 尝试还原立方体结构
                S = int(round(N ** (1 / 3)))

                # 只有当 N 是完美的立方体数时才做卷积
                if S ** 3 == N:
                    x_img = x.transpose(1, 2).view(B, C, S, S, S)
                    local_out = self.local_branch(x_img)
                    local_out = local_out.flatten(2).transpose(1, 2)
                else:
                    return original_out

            if local_out is not None:
                return original_out + self.alpha * local_out
            else:
                return original_out

        except Exception as e:
            # 兜底：出错时不中断训练
            return original_out


def inject_parallel_innovation(model):
    """
    🔥 注入函数：
    只在浅层 (Channel <= 192) 注入增强版 MLP，深层跳过。
    """
    print("💉 正在向 Swin Backbone 注入并行 3D 卷积模块 (Standard Conv Mode)...")

    count_replaced = 0
    count_skipped = 0

    # 设定浅层的阈值。Feature Size=96时：
    # Stage 0 = 96, Stage 1 = 192 (浅层 -> 注入)
    # Stage 2 = 384, Stage 3 = 768 (深层 -> 跳过)
    SHALLOW_THRESHOLD = 192

    for name, module in model.swin_unetr.named_modules():
        if "SwinTransformerBlock" in module.__class__.__name__:
            if hasattr(module, "mlp"):
                old_mlp = module.mlp
                if isinstance(old_mlp, ParallelLocalMLP): continue

                # 获取维度
                dim = 0
                if hasattr(old_mlp, "linear1"):
                    dim = old_mlp.linear1.in_features
                elif hasattr(old_mlp, "fc1"):
                    dim = old_mlp.fc1.in_features

                if dim == 0: continue

                # 🔥 策略：只在浅层注入
                if dim <= SHALLOW_THRESHOLD:
                    new_mlp = ParallelLocalMLP(old_mlp, dim)
                    module.mlp = new_mlp
                    count_replaced += 1
                else:
                    count_skipped += 1

    print(f"✅ 注入完成: {count_replaced} 个浅层模块被替换 (Standard Conv, Alpha=0.5)")
    print(f"⏩ 跳过处理: {count_skipped} 个深层模块保持原样")

    return model


# =========================================================================
# 3. 纯净融合模块 (SE Only)
# =========================================================================
class SEAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y.expand_as(x)


class FeatureFusionBlock(nn.Module):
    def __init__(self, swin_channels, cnn_channels):
        super().__init__()
        if cnn_channels != swin_channels:
            self.proj = nn.Conv3d(cnn_channels, swin_channels, kernel_size=1, bias=False)
        else:
            self.proj = nn.Identity()

        self.norm = nn.InstanceNorm3d(swin_channels)
        self.act = nn.LeakyReLU()

        self.se_block = SEAttentionBlock(swin_channels)

    def forward(self, swin_feat, cnn_feat):
        cnn_proj = self.proj(cnn_feat)
        cnn_proj = self.norm(cnn_proj)
        cnn_proj = self.act(cnn_proj)

        fused = swin_feat + cnn_proj
        fused = self.se_block(fused)
        return fused


# =========================================================================
# 4. MAIN MODEL
# =========================================================================
class SwinUNETR_Optimized(nn.Module):
    def __init__(self,
                 img_size=(64, 64, 64),
                 in_channels=1,
                 out_channels=1,
                 feature_size=96,  # 🔥 96
                 use_checkpoint=True,

                 ):
        super().__init__()

        print(f"🔧 初始化 Optimized Model (V10.3 High-Power) | Feature Size: {feature_size}")
        print("⚡ 配置: Swin(96) + StandardConv-MLP(Shallow) + SE + Wavelet")

        self.swin_unetr = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24)
        )

        # 🔥 注入创新点 (只注入浅层)
        inject_parallel_innovation(self)

        self.cnn_encoder = SimpleCNNEncoder3D(in_channels=in_channels, base_filters=24)

        self.fusion0 = FeatureFusionBlock(swin_channels=feature_size, cnn_channels=24)
        self.fusion1 = FeatureFusionBlock(swin_channels=feature_size * 2, cnn_channels=48)
        self.fusion2 = FeatureFusionBlock(swin_channels=feature_size * 4, cnn_channels=96)
        self.fusion3 = FeatureFusionBlock(swin_channels=feature_size * 8, cnn_channels=192)

        bottleneck_dim = feature_size * 16
        enc4_dim = feature_size * 8
        enc3_dim = feature_size * 4

        self.wavelet_dec5 = WaveletUnetrUpBlock(
            spatial_dims=3, in_channels=bottleneck_dim, out_channels=enc4_dim,
            kernel_size=3, upsample_kernel_size=2
        )

        self.ds_head1 = UnetOutBlock(spatial_dims=3, in_channels=enc3_dim, out_channels=out_channels)
        self.ds_head2 = UnetOutBlock(spatial_dims=3, in_channels=enc4_dim, out_channels=out_channels)

    def forward(self, x_in):
        swin_features = self.swin_unetr.swinViT(x_in, self.swin_unetr.normalize)
        cnn_features = self.cnn_encoder(x_in)

        fused_features = list(swin_features)
        fused_features[0] = self.fusion0(swin_features[0], cnn_features[0])
        fused_features[1] = self.fusion1(swin_features[1], cnn_features[1])
        fused_features[2] = self.fusion2(swin_features[2], cnn_features[2])
        fused_features[3] = self.fusion3(swin_features[3], cnn_features[3])

        enc0 = self.swin_unetr.encoder1(x_in)
        dec4 = self.swin_unetr.encoder10(fused_features[4])

        dec3 = self.wavelet_dec5(dec4, fused_features[3])
        out_ds2 = self.ds_head2(dec3)

        dec2 = self.swin_unetr.decoder4(dec3, fused_features[2])
        out_ds1 = self.ds_head1(dec2)

        dec1 = self.swin_unetr.decoder3(dec2, fused_features[1])
        dec0 = self.swin_unetr.decoder2(dec1, fused_features[0])
        out = self.swin_unetr.decoder1(dec0, enc0)

        logits = self.swin_unetr.out(out)

        if self.training:
            return [logits, out_ds1, out_ds2]
        else:
            return logits


# ================= 权重加载函数 =================
def load_pretrained_weights(model, weight_path):
    if not os.path.exists(weight_path):
        print("⚠️ No weights found.")
        return model

    print(f"⏳ Loading weights from {os.path.basename(weight_path)}...")
    try:
        checkpoint = torch.load(weight_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return model

    if "state_dict" in checkpoint:
        pretrained_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        pretrained_dict = checkpoint["model"]
    else:
        pretrained_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = {}
    matched = 0

    for k, v in pretrained_dict.items():
        key = k.replace("module.", "")
        target_keys = [key, "swin_unetr." + key, "swin_unetr.swinViT." + key]

        for target_k in target_keys:
            if target_k in model_dict:
                if v.shape == model_dict[target_k].shape:
                    new_state_dict[target_k] = v
                    matched += 1
                # 处理被我们替换后的 MLP 权重加载 (只加载主路，忽略支路)
                elif target_k.replace("mlp.", "mlp.original_mlp.") in model_dict:
                    new_target = target_k.replace("mlp.", "mlp.original_mlp.")
                    if v.shape == model_dict[new_target].shape:
                        new_state_dict[new_target] = v
                        matched += 1

    if matched > 0:
        model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Loaded {matched} layers.")
    else:
        print("❌ Warning: 0 layers loaded! (Safe for 96 random init)")

    return model