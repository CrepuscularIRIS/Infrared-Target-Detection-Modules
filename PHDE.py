import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.conv import Conv

class DepthwiseLaplacian(nn.Module):
    """
    Applies a fixed 3x3 depthwise Laplacian kernel.
    Assumes input and output channels are the same.
    Padding is set to 'same' to preserve spatial dimensions.
    """
    def __init__(self, ch, kernel_size=3):
        super().__init__()
        # Standard 3x3 Laplacian kernel [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        # Other variants like [[1, 1, 1], [1, -8, 1], [1, 1, 1]] could also be used.
        # We'll use the simpler [-4] center version.
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32)

        # Expand dimensions to [out_channels, in_channels/groups, kH, kW]
        # For depthwise, out_channels = in_channels = groups = ch
        laplacian_kernel = laplacian_kernel.view(1, 1, kernel_size, kernel_size)
        laplacian_kernel = laplacian_kernel.repeat(ch, 1, 1, 1) # Repeat for each channel

        self.conv = nn.Conv2d(
            ch, ch, kernel_size=kernel_size,
            stride=1, padding=kernel_size // 2,
            groups=ch, # Depthwise convolution
            bias=False
        )

        # Set the fixed kernel and make it non-trainable
        self.conv.weight.data = laplacian_kernel
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)

class PHDE(nn.Module):
    """
    Persistent Heat-Diffusion Position Encoding (PHDE) module.

    Integrates Persistent Heat Kernel Signatures (PHKS) derived from
    Topological Data Analysis (TDA) with learnable weights within a
    convolutional network framework. Aims to achieve multi-scale robustness
    and noise suppression for small target localization.

    Approximates heat diffusion using a truncated Taylor expansion of the
    matrix exponential involving the graph Laplacian, implemented efficiently
    with depthwise convolutions. Learns adaptive weights for different diffusion
    times based on global context.

    Args:
        c (int): Number of input channels.
        times (tuple[float]): A tuple of initial diffusion times (t).
                               Defaults to (0.5, 1.0, 2.0).
        r (int): Channel reduction ratio for the initial 1x1 convolution.
                 Defaults to 4.
        k (int): Kernel size for the depthwise Laplacian approximation.
                 Defaults to 3. Must be odd.
    """
    def __init__(self, c: int, times: tuple[float] = (0.5, 1.0, 2.0), r: int = 4, k: int = 3):
        super().__init__()
        if not isinstance(times, (list, tuple)) or len(times) < 2:
            raise ValueError("`times` must be a sequence of at least two values.")
        if k % 2 == 0:
            raise ValueError("`k` (kernel_size for Laplacian) must be odd.")

        self.times = tuple(times)
        num_times = len(times)

        # Calculate reduced channels, ensuring a minimum value (e.g., 8)
        cr = max(8, c // r)

        # 1x1 Convolution for channel reduction
        self.reduce = nn.Conv2d(c, cr, kernel_size=1, stride=1, padding=0, bias=False)

        # Depthwise Laplacian operator (fixed kernel)
        # Approximates L in the heat equation e^{-tL}
        self.dw_lap = DepthwiseLaplacian(cr, kernel_size=k)

        # Learnable diffusion time scales (initialized from `times`)
        # Allows fine-tuning of the diffusion process
        self.t_scale = nn.Parameter(torch.tensor(list(self.times)), requires_grad=True)

        # Adaptive Weight Prediction Branch
        # Predicts coefficients for combining persistent differences based on global context.
        self.pool = nn.AdaptiveAvgPool2d(1) # Global Average Pooling
        self.w_pred = nn.Sequential(
            nn.Conv2d(cr, max(8, cr // 2), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            # Output channels = number of diffusion times T
            nn.Conv2d(max(8, cr // 2), num_times, kernel_size=1, bias=False)
        )

        # Fusion Layer
        # Combines the weighted persistent difference map into a single attention map.
        # Note: The input channels here are derived from the processing in forward().
        # Based on the forward pass logic (mean across C', sum across T-1), the input
        # to the Conv2d should be 1. The provided __init__ snippet had num_times,
        # which seems inconsistent with the forward pass snippet. Assuming 1 is correct.
        self.fuse = nn.Sequential(
            # Input channel is 1 after summing weighted persistence differences
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid() # Output attention map in range [0, 1]
        )

    def heat_step(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Applies one step of heat diffusion using the 2nd-order Taylor approximation:
        e^{-tL} f ≈ (I - tL + t^2 L^2 / 2) f
        where L is approximated by self.dw_lap.

        Args:
            x (torch.Tensor): Input feature map (B x C' x H x W).
            t (torch.Tensor): Diffusion time (scalar tensor).

        Returns:
            torch.Tensor: Feature map after heat diffusion approximation.
        """
        # Calculate L*x
        y1 = self.dw_lap(x)
        # Calculate L*(L*x) = L^2*x
        y2 = self.dw_lap(y1)

        # Ensure t has compatible shape for broadcasting if needed (though usually scalar here)
        t = t.view(1, 1, 1, 1) # Reshape scalar t for broadcasting with BxCxHxW tensors

        # Apply Taylor expansion: x - t*L*x + 0.5*t^2*L^2*x
        return x - t * y1 + 0.5 * (t * t) * y2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PHDE module.

        Args:
            x (torch.Tensor): Input feature map (B x C x H x W).

        Returns:
            torch.Tensor: Output feature map with positional encoding applied
                          (B x C x H x W).
        """
        b, c, h, w = x.shape

        # 1. Reduce channels
        y = self.reduce(x) # B x Cr x H x W

        # 2. Compute heat diffusion at different time scales
        feats = []
        # Ensure t_scale values remain non-negative during training (optional, but good practice)
        # Using F.relu or torch.clamp(min=0) on self.t_scale can enforce this.
        # Here we use the raw parameter values, assuming they stay reasonable.
        current_times = self.t_scale
        for i in range(len(current_times)):
            feats.append(self.heat_step(y, current_times[i]))

        # Stack features along a new 'time' dimension
        H = torch.stack(feats, dim=1) # B x T x Cr x H x W (where T = len(times))

        # 3. Calculate Persistent Differences (approximating persistence barcode intervals)
        # P_i = H(t_{i+1}) - H(t_i)
        P = H[:, 1:, ...] - H[:, :-1, ...] # B x (T-1) x Cr x H x W

        # 4. Average persistence differences across the reduced channel dimension (Cr)
        # This reduces parameters and focuses on spatial patterns.
        P = P.mean(dim=2) # B x (T-1) x H x W

        # 5. Predict Adaptive Weights
        # Use the initial reduced feature map 'y' for global context
        w_ctx = self.pool(y)      # B x Cr x 1 x 1
        w = self.w_pred(w_ctx)    # B x T x 1 x 1
        w = F.softmax(w, dim=1)   # Normalize weights across time scales (B x T x 1 x 1)

        # Select weights corresponding to the differences P_i (needs T-1 weights)
        # We use the weights w_i for the interval [t_i, t_{i+1}], so we need w_0 to w_{T-2}
        w = w[:, :-1, ...] # B x (T-1) x 1 x 1

        # 6. Apply weights and Fuse
        # Weight the persistent differences and sum them up
        # (P * w) -> B x (T-1) x H x W (element-wise multiplication with broadcasting)
        # .sum(dim=1, keepdim=True) -> B x 1 x H x W
        weighted_P_sum = (P * w).sum(dim=1, keepdim=True)

        # Apply the fusion layer (Conv2d 1x1 -> Sigmoid)
        a = self.fuse(weighted_P_sum) # B x 1 x H x W (Attention map)

        # 7. Apply attention map to original input using residual connection
        # x * (1 + a) enhances features based on the persistent heat signature
        return x * (1.0 + a)


# Example Usage:
if __name__ == '__main__':
    # Example parameters
    input_channels = 64
    batch_size = 4
    height, width = 32, 32

    # Instantiate the module
    phde_module = PHDE(c=input_channels, times=(0.4, 1.0, 2.2), r=4, k=3)
    # print(phde_module)

    # Create a dummy input tensor
    dummy_input = torch.randn(batch_size, input_channels, height, width)

    # Forward pass
    output = phde_module(dummy_input)

    # Check output shape
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Calculate number of parameters (optional)
    num_params = sum(p.numel() for p in phde_module.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in PHDE: {num_params}")

    # --- Test specific parts ---
    # Test heat_step
    phde_module.eval() # Set to eval mode if using dropout/batchnorm layers inside (not here)
    reduced_input = phde_module.reduce(dummy_input)
    time_step_output = phde_module.heat_step(reduced_input, phde_module.t_scale[0])
    print(f"Shape after reduce: {reduced_input.shape}")
    print(f"Shape after one heat_step: {time_step_output.shape}")

    # Verify t_scale parameter exists and is learnable
    print(f"t_scale parameter: {phde_module.t_scale}")
    print(f"t_scale requires_grad: {phde_module.t_scale.requires_grad}")

# ------------- New PHDEConv Module -------------
# Conv already imported above

class PHDEConv(nn.Module):
    """
    Convolutional layer combined with Persistent Heat-Diffusion Position Encoding.

    Applies a standard convolution (Conv), followed by the PHDE module.
    The PHDE module enhances feature representations by encoding heat diffusion information,
    particularly useful for capturing multi-scale features and small targets.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        k (int): Kernel size for the convolution. Defaults to 1.
        s (int): Stride for the convolution. Defaults to 1.
        p (int, optional): Padding for the convolution. Defaults to None (autopad).
        g (int): Number of groups for the convolution. Defaults to 1.
        d (int): Dilation for the convolution. Defaults to 1.
        act (bool | nn.Module): Activation function. Defaults to True (use default).
        bias (bool): Whether to use bias in the convolution. Defaults to False.
        phde_times (tuple[float]): A tuple of initial diffusion times (t) for PHDE. Defaults to (0.5, 1.0, 2.0).
        phde_r (int): Channel reduction ratio for PHDE. Defaults to 4.
        phde_k (int): Kernel size for the depthwise Laplacian approximation in PHDE. Defaults to 3.
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, 
                 phde_times=(0.5, 1.0, 2.0), phde_r=4, phde_k=3):
        super().__init__()
        # Standard convolution layer
        self.conv = Conv(c1, c2, k, s, p=p, g=g, d=d, act=act)
        # PHDE module applied on the output of the convolution (c2 channels)
        self.phde = PHDE(c=c2, times=phde_times, r=phde_r, k=phde_k)

    def forward(self, x):
        """Applies Conv -> PHDE to the input tensor."""
        y = self.conv(x)
        return self.phde(y)

    def forward_fuse(self, x):
        """
        Forward pass for fused model. 
        Applies the fused Conv and then the PHDE module.
        """
        y = self.conv.forward_fuse(x)
        return self.phde(y)

class C2fPHDE(nn.Module):
    """
    C2f module with PHDE (Persistent Heat-Diffusion Position Encoding).
    
    This module combines the faster implementation of CSP Bottleneck with PHDE
    for enhanced multi-scale feature extraction, especially beneficial for small targets.
    
    Args:
        c1 (int): Number of input channels
        c2 (int): Number of output channels
        n (int): Number of bottlenecks in the CSP layer. Default is 1
        shortcut (bool): Whether to use shortcut connections. Default is False
        g (int): Number of groups for conv. Default is 1
        e (float): Expansion ratio. Default is 0.5
        phde_times (tuple): Diffusion times for PHDE. Default is (0.5, 1.0, 2.0)
        phde_r (int): Channel reduction ratio for PHDE. Default is 4
        phde_k (int): Kernel size for Laplacian in PHDE. Default is 3
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, 
                 phde_times=(0.5, 1.0, 2.0), phde_r=4, phde_k=3):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.phde = PHDE(c=c2, times=phde_times, r=phde_r, k=phde_k)

    def forward(self, x):
        """Forward pass through the C2fPHDE module."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        # Apply PHDE to the output
        return self.phde(out)

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        # Apply PHDE to the output
        return self.phde(out)

class PHDESPPF(nn.Module):
    """
    PHDESPPF: Persistent Heat-Diffusion Encoding Spatial Pyramid Pooling Fast
    
    将PHDE注意力机制与SPPF结构相结合，提升特征表示能力。
    遵循原始SPPF的设计，但在输出前增加PHDE注意力机制，有利于捕获多尺度特征和提高小目标检测能力。
    
    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        k (int): 最大池化的核大小，默认为5
        phde_times (tuple): PHDE的扩散时间序列，默认为(0.5, 1.0, 2.0)
        phde_r (int): PHDE的通道压缩比例，默认为4
        phde_k (int): PHDE中Laplacian算子的核大小，默认为3
    """
    
    def __init__(self, c1, c2, k=5, phde_times=(0.5, 1.0, 2.0), phde_r=4, phde_k=3):
        """
        初始化PHDESPPF模块
        
        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 最大池化的核大小，默认为5
            phde_times (tuple): PHDE的扩散时间序列
            phde_r (int): PHDE的通道压缩比例
            phde_k (int): PHDE中Laplacian算子的核大小
        """
        super().__init__()
        c_ = c1 // 2  # 隐藏通道数
        
        # 初始化层
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个1x1卷积，减少通道数
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # 最大池化层
        self.cv2 = Conv(c_ * 4, c2, 1, 1)  # 第二个1x1卷积，调整通道数到c2
        
        # PHDE注意力机制
        self.phde = PHDE(c=c2, times=phde_times, r=phde_r, k=phde_k)
    
    def forward(self, x):
        """前向传播"""
        # 第一步: 通过1x1卷积减少通道数
        x = self.cv1(x)  # [B, c1] -> [B, c_]
        
        # 第二步: 创建特征金字塔（与原始SPPF完全相同）
        y = [x]  # 存储金字塔各层特征
        y.extend(self.m(y[-1]) for _ in range(3))  # 应用三次最大池化
        
        # 第三步: 拼接所有特征并通过1x1卷积
        x = self.cv2(torch.cat(y, 1))  # [B, c_*4] -> [B, c2]
        
        # 第四步: 应用PHDE注意力机制增强特征表示
        x = self.phde(x)  # 应用PHDE注意力，保持输出通道数不变
        
        return x
