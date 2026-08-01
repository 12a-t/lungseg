import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
from monai.networks.blocks import UnetOutBlock, UnetrUpBlock
import os


# =============================================================================
# 1. IDWT and DWT Modules
# =============================================================================
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


class StandardDWTDownSample3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt_sim = nn.Conv3d(
            in_channels, in_channels * 8, kernel_size=2, stride=2, groups=in_channels, bias=False
        )
        out_half = out_channels // 2
        self.compress_low = nn.Conv3d(in_channels, out_half, kernel_size=1)
        self.compress_high = nn.Conv3d(in_channels * 7, out_half, kernel_size=1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        dwt_output = self.dwt_sim(x)
        C = dwt_output.shape[1] // 8
        low_freq = dwt_output[:, :C, ...]
        high_freqs = dwt_output[:, C:, ...]
        low_compressed = self.compress_low(low_freq)
        high_compressed = self.compress_high(high_freqs)
        combined = torch.cat([low_compressed, high_compressed], dim=1)
        return self.act(self.norm(combined))


class GGODWTDownSample3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt_sim = nn.Conv3d(
            in_channels, in_channels * 8, kernel_size=2, stride=2, groups=in_channels, bias=False
        )
        self.local_pool = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)
        self.th_mean = nn.Parameter(torch.tensor(0.25))
        self.scale_mean = nn.Parameter(torch.tensor(10.0))
        self.th_std = nn.Parameter(torch.tensor(0.08))
        self.scale_std = nn.Parameter(torch.tensor(15.0))

        out_half = out_channels // 2
        self.compress_low = nn.Conv3d(in_channels, out_half, kernel_size=1)
        self.compress_high = nn.Conv3d(in_channels * 7, out_half, kernel_size=1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        dwt_output = self.dwt_sim(x)
        B, C8, D, H, W = dwt_output.shape
        C = C8 // 8
        low_freq = dwt_output[:, :C, ...]
        high_freqs = dwt_output[:, C:, ...]

        mu = self.local_pool(low_freq)
        mu_sq = self.local_pool(low_freq ** 2)
        std = torch.sqrt(torch.relu(mu_sq - mu ** 2) + 1e-6)
        w_mean = torch.sigmoid((mu - self.th_mean) * self.scale_mean)
        w_std = torch.sigmoid((self.th_std - std) * self.scale_std)
        ggo_weight = w_mean * w_std

        low_freq_enhanced = low_freq * (1.0 + ggo_weight)

        low_compressed = self.compress_low(low_freq_enhanced)
        high_compressed = self.compress_high(high_freqs)
        combined = torch.cat([low_compressed, high_compressed], dim=1)
        return self.act(self.norm(combined))


class WaveletUnetrUpBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, kernel_size=3, upsample_kernel_size=2,
                 norm_name="instance", res_block=True):
        super().__init__()
        self.wavelet_generator = nn.Conv3d(in_channels, out_channels * 8, kernel_size=1, bias=False)
        self.idwt = IDWT3D(upscale_factor=2)
        self.conv_block = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, upsample_kernel_size=upsample_kernel_size, norm_name=norm_name,
            res_block=res_block,
        ).conv_block

    def forward(self, inp, skip):
        subbands = self.wavelet_generator(inp)
        upsampled = self.idwt(subbands)
        if upsampled.shape[2:] != skip.shape[2:]:
            upsampled = F.interpolate(upsampled, size=skip.shape[2:], mode='trilinear', align_corners=False)
        concat = torch.cat((upsampled, skip), dim=1)
        return self.conv_block(concat)


# =============================================================================
# 2. CNN Encoder
# =============================================================================
class SimpleCNNEncoder3D(nn.Module):
    def __init__(self, in_channels=1, base_filters=24):
        super().__init__()
        self.down0 = GGODWTDownSample3d(in_channels, base_filters)
        self.down1 = StandardDWTDownSample3d(base_filters, base_filters * 2)
        self.down2 = StandardDWTDownSample3d(base_filters * 2, base_filters * 4)
        self.down3 = StandardDWTDownSample3d(base_filters * 4, base_filters * 8)
        self.down4 = StandardDWTDownSample3d(base_filters * 8, base_filters * 16)

    def forward(self, x):
        c0 = self.down0(x)
        c1 = self.down1(c0)
        c2 = self.down2(c1)
        c3 = self.down3(c2)
        c4 = self.down4(c3)
        return [c0, c1, c2, c3, c4]


# =============================================================================
# 3. Structure-Aware Parallel Module
# =============================================================================
class ParallelLocalMLP(nn.Module):
    def __init__(self, original_mlp, dim):
        super().__init__()
        self.original_mlp = original_mlp

        self.local_branch = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=1, bias=False),
            nn.InstanceNorm3d(dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=1, bias=False)
        )

        laplace_kernel = torch.zeros(3, 3, 3)
        laplace_kernel[1, 1, 1] = 6.0
        laplace_kernel[0, 1, 1] = -1.0
        laplace_kernel[2, 1, 1] = -1.0
        laplace_kernel[1, 0, 1] = -1.0
        laplace_kernel[1, 2, 1] = -1.0
        laplace_kernel[1, 1, 0] = -1.0
        laplace_kernel[1, 1, 2] = -1.0
        kernel_3d = laplace_kernel.view(1, 1, 3, 3, 3).repeat(dim, 1, 1, 1, 1)
        self.register_buffer('laplace_weight', kernel_3d)

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def _compute_complexity(self, x_img):
        L1 = torch.abs(F.conv3d(x_img, self.laplace_weight, padding=1, dilation=1, groups=x_img.size(1)))
        L2 = torch.abs(F.conv3d(x_img, self.laplace_weight, padding=2, dilation=2, groups=x_img.size(1)))
        L3 = torch.abs(F.conv3d(x_img, self.laplace_weight, padding=3, dilation=3, groups=x_img.size(1)))

        L_max = torch.max(torch.max(L1, L2), L3)
        spatial_L = L_max.mean(dim=1, keepdim=True)

        eps = 1e-5
        mean = spatial_L.mean(dim=[2, 3, 4], keepdim=True)
        std = spatial_L.std(dim=[2, 3, 4], keepdim=True)
        spatial_L_norm = (spatial_L - mean) / (std + eps)

        complexity_map = torch.sigmoid(spatial_L_norm * self.temperature) * 2.0
        return complexity_map

    def forward(self, x):
        original_out = self.original_mlp(x)

        if x.ndim == 5:
            is_channel_last = (x.shape[1] != self.local_branch[0].in_channels)
            x_img = x.permute(0, 4, 1, 2, 3) if is_channel_last else x

            out_local = self.local_branch(x_img)
            complexity_map = self._compute_complexity(x_img)
            local_out = out_local * complexity_map

            if is_channel_last:
                local_out = local_out.permute(0, 2, 3, 4, 1)

            return original_out + self.alpha * local_out

        elif x.ndim == 3:
            B, N, C = x.shape
            S = int(round(N ** (1 / 3)))
            if S ** 3 == N:
                x_img = x.transpose(1, 2).view(B, C, S, S, S)
                out_local = self.local_branch(x_img)
                complexity_map = self._compute_complexity(x_img)
                out_local = out_local * complexity_map
                local_out = out_local.flatten(2).transpose(1, 2)
                return original_out + self.alpha * local_out
            else:
                return original_out
        else:
            return original_out


def inject_parallel_module(model):
    count_replaced = 0
    SHALLOW_THRESHOLD = 192

    for name, module in model.swin_unetr.named_modules():
        if "SwinTransformerBlock" in module.__class__.__name__ and hasattr(module, "mlp"):
            old_mlp = module.mlp
            if isinstance(old_mlp, ParallelLocalMLP):
                continue

            dim = 0
            if hasattr(old_mlp, "linear1"):
                dim = old_mlp.linear1.in_features
            elif hasattr(old_mlp, "fc1"):
                dim = old_mlp.fc1.in_features

            if dim > 0 and dim <= SHALLOW_THRESHOLD:
                module.mlp = ParallelLocalMLP(old_mlp, dim)
                count_replaced += 1

    return model


# =============================================================================
# 4. Asymmetric Dual-Stream Feature Fusion (Multi-Head Bidirectional Attention)
# =============================================================================
class BidirectionalChannelCrossAttention3D(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_s = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)
        self.k_s = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)
        self.v_s = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)

        self.q_c = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)
        self.k_c = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)
        self.v_c = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)

        self.out_proj_s = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels)
        )
        self.out_proj_c = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels)
        )
        self.act = nn.GELU()

    def forward(self, swin, cnn):
        B, C, D, H, W = swin.shape
        N = D * H * W

        qs = self.q_s(swin).view(B, self.num_heads, self.head_dim, N)
        ks = self.k_s(swin).view(B, self.num_heads, self.head_dim, N)
        vs = self.v_s(swin).view(B, self.num_heads, self.head_dim, N)

        qc = self.q_c(cnn).view(B, self.num_heads, self.head_dim, N)
        kc = self.k_c(cnn).view(B, self.num_heads, self.head_dim, N)
        vc = self.v_c(cnn).view(B, self.num_heads, self.head_dim, N)

        attn_s2c = (qs @ kc.transpose(-2, -1)) * self.scale
        attn_s2c = attn_s2c.softmax(dim=-1)
        out_s = (attn_s2c @ vc).view(B, C, D, H, W)
        out_s = self.act(self.out_proj_s(out_s))

        attn_c2s = (qc @ ks.transpose(-2, -1)) * self.scale
        attn_c2s = attn_c2s.softmax(dim=-1)
        out_c = (attn_c2s @ vs).view(B, C, D, H, W)
        out_c = self.act(self.out_proj_c(out_c))

        return swin + out_s, cnn + out_c


class CrossAttentionFusionBlock(nn.Module):
    def __init__(self, swin_channels, cnn_channels, num_heads=4):
        super().__init__()
        self.proj = nn.Conv3d(cnn_channels, swin_channels, kernel_size=1, bias=False) if cnn_channels != swin_channels else nn.Identity()
        self.norm = nn.InstanceNorm3d(swin_channels)
        self.act = nn.GELU()
        self.cross_attn = BidirectionalChannelCrossAttention3D(swin_channels, num_heads=num_heads)

    def forward(self, swin_feat, cnn_feat):
        cnn_proj = self.act(self.norm(self.proj(cnn_feat)))
        swin_out, _ = self.cross_attn(swin_feat, cnn_proj)
        return swin_out


# =============================================================================
# 5. DSSW-Net: Main Model
# =============================================================================
class DSSWNet(nn.Module):
    def __init__(self, img_size=(64, 64, 64), in_channels=1, out_channels=1, feature_size=96, use_checkpoint=True):
        super().__init__()

        self.swin_unetr = SwinUNETR(
            in_channels=in_channels, out_channels=out_channels, feature_size=feature_size,
            use_checkpoint=use_checkpoint, spatial_dims=3, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24)
        )

        self.swin_unetr = inject_parallel_module(self)

        self.cnn_encoder = SimpleCNNEncoder3D(in_channels=in_channels, base_filters=24)

        self.fusion0 = CrossAttentionFusionBlock(swin_channels=feature_size, cnn_channels=24, num_heads=4)
        self.fusion1 = CrossAttentionFusionBlock(swin_channels=feature_size * 2, cnn_channels=48, num_heads=4)
        self.fusion2 = CrossAttentionFusionBlock(swin_channels=feature_size * 4, cnn_channels=96, num_heads=4)
        self.fusion3 = CrossAttentionFusionBlock(swin_channels=feature_size * 8, cnn_channels=192, num_heads=4)
        self.fusion4 = CrossAttentionFusionBlock(swin_channels=feature_size * 16, cnn_channels=384, num_heads=4)

        bottleneck_dim = feature_size * 16
        enc4_dim = feature_size * 8
        enc3_dim = feature_size * 4

        self.wavelet_dec5 = WaveletUnetrUpBlock(
            spatial_dims=3, in_channels=bottleneck_dim, out_channels=enc4_dim, kernel_size=3, upsample_kernel_size=2
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
        fused_features[4] = self.fusion4(swin_features[4], cnn_features[4])

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


def load_pretrained_weights(model, weight_path):
    if not os.path.exists(weight_path):
        return model

    try:
        checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Load error: {e}")
        return model

    pretrained_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    model_dict = model.state_dict()
    new_state_dict = {}
    matched = 0

    for k, v in pretrained_dict.items():
        key = k.replace("module.", "")
        target_keys = [key, "swin_unetr." + key, "swin_unetr.swinViT." + key]
        for target_k in target_keys:
            if target_k in model_dict and v.shape == model_dict[target_k].shape:
                new_state_dict[target_k] = v
                matched += 1
            elif target_k.replace("mlp.", "mlp.original_mlp.") in model_dict:
                new_target = target_k.replace("mlp.", "mlp.original_mlp.")
                if v.shape == model_dict[new_target].shape:
                    new_state_dict[new_target] = v
                    matched += 1

    if matched > 0:
        model.load_state_dict(new_state_dict, strict=False)

    return model