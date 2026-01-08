# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "CBAM",
    "CBAMLeaf",
    "ChannelAttention",
    "Concat",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "CoordAtt",
    "CoordAttPlus",
    "DeformConv",
    "DWConv",
    "DWConvTranspose2d",
    "ECA",
    "Focus",
    "GhostConv",
    "Index",
    "LightConv",
    "LSKA",
    "RepConv",
    "SpatialAttention",
    "SPDConv",

)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p




class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution (SPD-Conv)
    ストライド畳み込み（情報の損失を伴うダウンサンプリング）の代わりに使用します。
    空間次元をチャンネル次元に移動させることで、情報を保持したままダウンサンプリングを行います。
    
    Reference: "No More Strided Convolutions for CNNs"
    """
    def __init__(self, c1, c2, dimension=1):
        super().__init__()
        # Space-to-Depthによりチャンネル数は4倍になります (c1 * 4)
        # その後、目的のチャンネル数 c2 に圧縮するための畳み込み層を通します
        self.conv = Conv(c1 * 4, c2, 3, 1) # Kernel=3, Stride=1 (No downsampling in conv)

    def forward(self, x):
        # x: (B, C, H, W)
        # H, W 次元を2つずつスキップしてスライスし、チャンネル方向に結合します
        
        # [0::2, 0::2] -> 偶数行、偶数列
        # [1::2, 0::2] -> 奇数行、偶数列
        # [0::2, 1::2] -> 偶数行、奇数列
        # [1::2, 1::2] -> 奇数行、奇数列
        
        x = torch.cat([
            x[..., 0::2, 0::2],
            x[..., 1::2, 0::2],
            x[..., 0::2, 1::2],
            x[..., 1::2, 1::2]
        ], 1)
        
        # x becomes (B, C*4, H/2, W/2)
        return self.conv(x)

class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Apply activation and convolution transpose operation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, W, H) and output shape is (B, 4C, W/2, H/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class CBAMLeaf(nn.Module):
    """Enhanced CBAM specifically optimized for leaf disease detection.
    
    This is a custom CBAM variant tailored for leaf detection tasks with:
    - Stronger channel attention (lower reduction ratio for more capacity)
    - Larger spatial attention kernel (9x9 for 1024px images)
    - Residual connections for better gradient flow
    - Optimized for detecting irregular leaf shapes and disease spots
    
    Improvements over standard CBAM:
    1. Channel Attention:
       - Reduction ratio: 16→8 (more powerful feature transformation)
       - Dual pooling: Average + Max pooling for richer features
    
    2. Spatial Attention:
       - Kernel size: 7→9 (larger receptive field for 1024px images)
       - Better capture of leaf boundaries and disease spot patterns
    
    3. Residual Connection:
       - Learnable alpha parameter for attention/identity balance
       - Prevents over-suppression of features
    
    Perfect for:
    - Irregular leaf edge detection
    - Overlapping leaf separation
    - Multi-scale disease spot detection
    - Color/texture + spatial location joint modeling
    
    Attributes:
        channel_attention (ChannelAttentionLeaf): Enhanced channel attention.
        spatial_attention (SpatialAttentionLeaf): Enhanced spatial attention.
        alpha (nn.Parameter): Learnable residual weight.
    
    References:
        Base CBAM: Woo et al. "CBAM: Convolutional Block Attention Module"
        https://arxiv.org/abs/1807.06521
        
        Enhanced for leaf detection with architecture optimizations.
    
    Examples:
        >>> cbam_leaf = CBAMLeaf(256)
        >>> x = torch.randn(1, 256, 128, 128)
        >>> out = cbam_leaf(x)  # Enhanced attention with residuals
    """
    
    def __init__(self, c1, reduction_ratio=8, spatial_kernel=9):
        """Initialize CBAMLeaf with enhanced parameters.
        
        Args:
            c1 (int): Number of input channels.
            reduction_ratio (int): Channel reduction ratio (default: 8, stronger than CBAM's 16).
            spatial_kernel (int): Spatial attention kernel size (default: 9, larger than CBAM's 7).
        """
        super().__init__()
        
        # Enhanced Channel Attention
        self.channel_attention = ChannelAttentionLeaf(c1, reduction_ratio=reduction_ratio)
        
        # Enhanced Spatial Attention  
        self.spatial_attention = SpatialAttentionLeaf(kernel_size=spatial_kernel)
        
        # Learnable residual weight
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        """Apply enhanced channel and spatial attention with residual connection.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Attended output with residual connection.
        """
        identity = x
        
        # Apply channel attention
        x = self.channel_attention(x)
        
        # Apply spatial attention
        x = self.spatial_attention(x)
        
        # Residual connection with learnable weight
        # out = α * attention + (1-α) * identity
        alpha = torch.sigmoid(self.alpha)  # Ensure 0 < alpha < 1
        out = alpha * x + (1 - alpha) * identity
        
        return out


class ChannelAttentionLeaf(nn.Module):
    """Enhanced Channel Attention for leaf detection.
    
    Stronger than standard ChannelAttention with:
    - Lower reduction ratio (more capacity)
    - Dual pooling (avg + max) fusion
    """
    
    def __init__(self, channels, reduction_ratio=8):
        """Initialize enhanced channel attention.
        
        Args:
            channels (int): Number of input channels.
            reduction_ratio (int): Reduction ratio (default: 8).
        """
        super().__init__()
        
        reduced_channels = max(channels // reduction_ratio, 8)
        
        # Shared MLP for both pooling paths
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Apply enhanced channel attention.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Channel-attended output.
        """
        # Dual pooling
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        # Combine and activate
        attention = self.sigmoid(avg_out + max_out)
        
        return x * attention


class SpatialAttentionLeaf(nn.Module):
    """Enhanced Spatial Attention for leaf detection.
    
    Larger kernel size for better spatial context on 1024px images.
    """
    
    def __init__(self, kernel_size=9):
        """Initialize enhanced spatial attention.
        
        Args:
            kernel_size (int): Convolution kernel size (default: 9).
        """
        super().__init__()
        
        assert kernel_size in [7, 9, 11], "kernel size must be 7, 9, or 11"
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Apply enhanced spatial attention.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Spatially-attended output.
        """
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        
        return x * attention


class Concat(nn.Module):
    """Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]


class LSKA(nn.Module):
    """Lightweight Spatial-Channel Attention Module.
    
    LSKA combines efficient spatial attention with channel attention using large-kernel convolutions
    decomposed into smaller kernels for computational efficiency. This module is particularly effective
    for detecting small and medium-sized objects.
    
    Attributes:
        conv0 (nn.Conv2d): Initial depthwise convolution.
        conv_spatial (nn.Conv2d): Spatial attention convolution.
        conv1 (nn.Conv2d): Final 1x1 convolution.
    
    References:
        https://arxiv.org/abs/2309.01439
    """
    
    def __init__(self, channels, kernel_size=23):
        """Initialize LSKA module.
        
        Args:
            channels (int): Number of input channels.
            kernel_size (int): Kernel size for spatial attention (default: 23).
        """
        super().__init__()
        # Decompose large kernel into smaller kernels for efficiency
        # k = 23 -> 7x7 depthwise + (1,7) + (7,1) + (1,11) + (11,1)
        self.conv0 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        
        # Large kernel decomposition for spatial attention
        self.conv_spatial_h = nn.Conv2d(channels, channels, (1, kernel_size), 
                                        padding=(0, kernel_size // 2), groups=channels, bias=False)
        self.conv_spatial_w = nn.Conv2d(channels, channels, (kernel_size, 1), 
                                        padding=(kernel_size // 2, 0), groups=channels, bias=False)
        
        # Channel attention
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        """Apply LSKA attention to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        
        Returns:
            (torch.Tensor): Attention-weighted output tensor.
        """
        # Spatial feature extraction with decomposed kernels
        u = x.clone()
        attn = self.conv0(x)
        
        # Large kernel spatial attention (decomposed)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_w(attn)
        
        # Channel attention
        attn = self.conv1(attn)
        attn = self.bn(attn)
        attn = self.act(attn)
        
        # Apply attention weights
        return u * attn


class DeformConv(nn.Module):
    """Deformable Convolution v2.
    
    Deformable convolution adapts its receptive field based on learned offsets, allowing it to
    model geometric transformations in the input feature map. This is particularly useful for
    detecting objects with irregular shapes or in overlapping scenarios.
    
    Attributes:
        offset_conv (nn.Conv2d): Convolution layer to generate offsets and modulation scalars.
        dcn (DeformConv2d): Torchvision deformable convolution layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function.
    
    References:
        Deformable ConvNets v2: https://arxiv.org/abs/1811.11168
    """
    
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        """Initialize DeformConv module.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups for convolution.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        try:
            from torchvision.ops import DeformConv2d
        except ImportError:
            raise ImportError(
                "DeformConv2d requires torchvision>=0.9.0. "
                "Install with: pip install torchvision>=0.9.0"
            )
        
        self.conv = Conv  # Store Conv class for compatibility
        p = autopad(k, p)
        
        # Offset conv: generates 2 * k * k offsets + k * k modulation scalars
        self.offset_conv = nn.Conv2d(
            c1, 
            3 * k * k,  # 2 * k * k for x,y offsets + k * k for modulation
            kernel_size=k,
            stride=s,
            padding=p,
            bias=True
        )
        
        # Initialize offset conv to zero (start with regular convolution)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        # Deformable convolution
        self.dcn = DeformConv2d(
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            bias=False
        )
        
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    def forward(self, x):
        """Apply deformable convolution to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        
        Returns:
            (torch.Tensor): Output tensor after deformable convolution.
        """
        # Generate offsets and modulation scalars
        out = self.offset_conv(x)
        
        # Split into offset (2*k*k) and mask (k*k)
        B, _, H, W = out.shape
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = torch.sigmoid(mask)
        
        # Apply deformable convolution
        try:
            x = self.dcn(x, offset, mask)
        except Exception as e:
            # Fallback to regular convolution if deformable conv fails
            import warnings
            warnings.warn(f"DeformConv failed, using regular conv: {e}")
            # Create a regular conv as fallback
            if not hasattr(self, 'fallback_conv'):
                self.fallback_conv = nn.Conv2d(
                    x.shape[1], self.dcn.out_channels,
                    kernel_size=self.dcn.kernel_size,
                    stride=self.dcn.stride,
                    padding=self.dcn.padding,
                    bias=False
                ).to(x.device)
            x = self.fallback_conv(x)
        
        x = self.bn(x)
        return self.act(x)


class CoordAtt(nn.Module):
    """Coordinate Attention for Efficient Mobile Network Design.
    
    Coordinate Attention factorizes channel attention into two 1D feature encoding processes
    that aggregate features along two spatial directions separately. This allows the attention
    module to capture long-range dependencies with precise positional information.
    
    Attributes:
        pool_h (nn.AdaptiveAvgPool2d): Pooling along height direction.
        pool_w (nn.AdaptiveAvgPool2d): Pooling along width direction.
        conv1 (nn.Conv2d): Shared convolution for dimensionality reduction.
        bn1 (nn.BatchNorm2d): Batch normalization.
        act (nn.SiLU): Activation function.
        conv_h (nn.Conv2d): Convolution for height attention.
        conv_w (nn.Conv2d): Convolution for width attention.
    
    References:
        https://arxiv.org/abs/2103.02907
    
    Examples:
        >>> attn = CoordAtt(256)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> out = attn(x)  # Output: (1, 256, 32, 32)
    """
    
    def __init__(self, channels, reduction=32):
        """Initialize Coordinate Attention module.
        
        Args:
            channels (int): Number of input channels.
            reduction (int): Channel reduction ratio (default: 32).
        """
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        # Intermediate channels
        mip = max(8, channels // reduction)
        
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=False)
    
    def forward(self, x):
        """Apply coordinate attention to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        
        Returns:
            (torch.Tensor): Output tensor with coordinate attention applied.
        """
        identity = x
        
        n, c, h, w = x.size()
        
        # X and Y direction pooling
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1) -> (B, C, 1, W) -> (B, C, W, 1)
        
        # Concatenate and transform  
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split back
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (B, C, W, 1) -> (B, C, 1, W)
        
        # Attention weights
        a_h = self.conv_h(x_h).sigmoid()  # (B, C, H, 1)
        a_w = self.conv_w(x_w).sigmoid()  # (B, C, 1, W)
        
        # Apply attention
        out = identity * a_h * a_w
        
        return out


class ECA(nn.Module):
    """Efficient Channel Attention module.
    
    ECA uses 1D convolution to model channel attention with minimal parameters,
    avoiding dimensionality reduction and capturing local cross-channel interaction.
    
    Attributes:
        avg_pool (nn.AdaptiveAvgPool2d): Global average pooling.
        conv (nn.Conv1d): 1D convolution for channel attention.
        sigmoid (nn.Sigmoid): Sigmoid activation.
    
    References:
        ECA-Net: Efficient Channel Attention for Deep CNNs
        https://arxiv.org/abs/1910.03151
    
    Examples:
        >>> eca = ECA(256)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> out = eca(x)  # Output: (1, 256, 32, 32)
    """
    
    def __init__(self, channels, k_size=3):
        """Initialize ECA module.
        
        Args:
            channels (int): Number of input channels.
            k_size (int): Kernel size for 1D convolution (default: 3).
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Apply ECA to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        
        Returns:
            (torch.Tensor): Output tensor with channel attention applied.
        """
        # Feature descriptor on the global spatial information
        y = self.avg_pool(x)  # (B, C, 1, 1)
        
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # Multi-scale information fusion
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)


class CoordAttPlus(nn.Module):
    """Enhanced Coordinate Attention with multi-scale pooling and deeper transform.
    
    Improvements over CoordAtt:
    - Multi-scale pooling: combines global and local spatial information
    - Deeper feature transformation: 2-layer network with more capacity
    - Residual connection: maintains gradient flow and feature preservation
    - Adaptive reduction: flexible channel reduction based on input size
    
    This enhanced version is particularly effective for:
    - Complex leaf edge detection with varying scales
    - Overlapping leaf boundary separation
    - Multi-scale disease spot detection
    
    Attributes:
        pool_h (nn.AdaptiveAvgPool2d): Global height-wise pooling.
        pool_w (nn.AdaptiveAvgPool2d): Global width-wise pooling.
        local_pool (nn.AvgPool2d): Local pooling for fine details.
        conv1 (nn.Conv2d): First transformation layer.
        bn1 (nn.BatchNorm2d): Batch normalization 1.
        conv2 (nn.Conv2d): Second transformation layer (deeper).
        bn2 (nn.BatchNorm2d): Batch normalization 2.
        act (nn.SiLU): Activation function.
        conv_h (nn.Conv2d): Height attention generation.
        conv_w (nn.Conv2d): Width attention generation.
        residual_weight (nn.Parameter): Learnable residual weight.
    
    References:
        Based on Coordinate Attention: https://arxiv.org/abs/2103.02907
        Enhanced with multi-scale and residual design
    
    Examples:
        >>> attn = CoordAttPlus(256, reduction=16)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> out = attn(x)  # Enhanced attention with multi-scale features
    """
    
    def __init__(self, channels, reduction=16):
        """Initialize CoordAttPlus module.
        
        Args:
            channels (int): Number of input channels.
            reduction (int): Channel reduction ratio (default: 16, more aggressive than CoordAtt's 32).
        """
        super().__init__()
        
        # Multi-scale pooling
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # Global height
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # Global width
        self.local_pool = nn.AvgPool2d(3, stride=1, padding=1)  # Local context
        
        # Adaptive intermediate channels
        mip = max(8, channels // reduction)
        
        # Deeper feature transformation (2 layers)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        
        # Attention generation
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Learnable residual weight
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        """Apply enhanced coordinate attention to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        
        Returns:
            (torch.Tensor): Output tensor with enhanced coordinate attention applied.
        """
        identity = x
        
        n, c, h, w = x.size()
        
        # Multi-scale feature extraction
        # Global spatial pooling
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1)
        
        # Local context (add local pooling information)
        x_local = self.local_pool(x)
        x_local_h = self.pool_h(x_local)
        x_local_w = self.pool_w(x_local).permute(0, 1, 3, 2)
        
        # Combine global and local (weighted sum)
        x_h = 0.7 * x_h + 0.3 * x_local_h  # Favor global but include local
        x_w = 0.7 * x_w + 0.3 * x_local_w
        
        # Concatenate H and W features
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        
        # Deeper feature transformation (2-layer network)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        y = self.conv2(y)  # Second layer for more capacity
        y = self.bn2(y)
        y = self.act(y)
        
        # Split back to H and W
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (B, C, 1, W)
        
        # Generate attention weights
        a_h = self.conv_h(x_h).sigmoid()  # (B, C, H, 1)
        a_w = self.conv_w(x_w).sigmoid()  # (B, C, 1, W)
        
        # Apply coordinate attention
        out_attn = identity * a_h * a_w
        
        # Residual connection with learnable weight
        # out = α * attention + (1-α) * identity
        alpha = torch.sigmoid(self.residual_weight)  # Ensure 0 < alpha < 1
        out = alpha * out_attn + (1 - alpha) * identity
        
        return out


