import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Helper function/class to compute Fractional Laplacian Kernel dynamically
class FractionalLaplacianKernel(nn.Module):
    """
    Computes the weights for a depthwise fractional Laplacian kernel based on
    the learnable order alpha.

    Kernel weights w_ij are proportional to ||p_ij||^-(dim + 2*alpha) for neighbors,
    and the center weight ensures the sum of weights is zero.
    Uses softplus on alpha for stability.
    """
    def __init__(self, k: int):
        super().__init__()
        if k % 2 == 0:
            raise ValueError(f"Kernel size k must be odd, but got {k}")
        self.k = k
        self.center = k // 2

        # Precompute relative coordinates and distances (squared) for efficiency
        coords = torch.arange(k)
        coords_x, coords_y = torch.meshgrid(coords, coords, indexing='ij')
        rel_coords_x = coords_x - self.center
        rel_coords_y = coords_y - self.center
        # Store distances, add epsilon for stability when calculating power
        # Store squared distance and compute sqrt later to avoid storing sqrt(0)
        squared_dist = rel_coords_x.float()**2 + rel_coords_y.float()**2
        # Register as buffer so it moves with the module (e.g., to GPU)
        self.register_buffer('squared_dist', squared_dist)
        self.register_buffer('identity_mask', (self.squared_dist == 0).float()) # Mask for center pixel

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Calculates the fractional Laplacian kernel weights.

        Args:
            alpha (torch.Tensor): The learnable fractional order (scalar tensor).

        Returns:
            torch.Tensor: The computed kernel weights of shape (1, 1, k, k).
        """
        # Ensure alpha is positive and numerically stable
        stable_alpha = F.softplus(alpha).clamp(min=0.1, max=2.0)  # 限制范围防止极值

        # Dimension d=2 for 2D images
        dim = 2
        exponent = -(dim + 2 * stable_alpha)

        # Calculate distance with better numerical stability
        dist = torch.sqrt(self.squared_dist + 1e-12) + 1e-8  # 双重保护

        # Calculate weights for neighbors with numerical stability
        # 使用log-exp技巧避免直接的负指数幂运算
        log_dist = torch.log(dist + 1e-12)
        log_weights = exponent * log_dist
        # 限制log权重范围防止溢出
        log_weights = torch.clamp(log_weights, min=-20, max=5)
        neighbor_weights = torch.exp(log_weights)

        # 检查并处理可能的NaN或inf
        neighbor_weights = torch.where(
            torch.isfinite(neighbor_weights), 
            neighbor_weights, 
            torch.zeros_like(neighbor_weights)
        )

        # Mask out the center pixel's contribution from the neighbor calculation
        neighbor_weights = neighbor_weights * (1.0 - self.identity_mask)

        # Set the center weight such that the sum of all weights is zero
        center_weight = -torch.sum(neighbor_weights)
        
        # 确保center_weight是有限的
        center_weight = torch.where(
            torch.isfinite(center_weight),
            center_weight,
            torch.tensor(-1.0, device=center_weight.device, dtype=center_weight.dtype)
        )

        # Combine center and neighbor weights
        kernel = neighbor_weights + center_weight * self.identity_mask

        # 最终检查并归一化
        kernel = torch.where(torch.isfinite(kernel), kernel, torch.zeros_like(kernel))
        
        # 简单归一化确保数值稳定
        kernel_abs_sum = torch.abs(kernel).sum()
        if kernel_abs_sum > 1e-12:
            kernel = kernel / kernel_abs_sum * 4.0  # 简单的归一化

        # Return kernel in the required shape (1, 1, k, k) for depthwise conv weight
        return kernel.unsqueeze(0).unsqueeze(0)

# Helper Module for Fixed Depthwise Laplacian
class DepthwiseLaplacian(nn.Module):
    """
    Applies a fixed depthwise Laplacian convolution using F.conv2d and buffer.
    Uses the standard 3x3 Laplacian kernel [0,1,0; 1,-4,1; 0,1,0]
    centered within a kxk kernel if k > 3.
    """
    def __init__(self, channels: int, k: int):
        super().__init__()
        if k % 2 == 0:
            raise ValueError(f"Kernel size k must be odd, but got {k}")
        self.k = k
        self.channels = channels
        self.padding = k // 2

        # Define the standard 3x3 Laplacian kernel (simpler version)
        laplacian_kernel_3x3 = torch.tensor([
            [0., 1., 0.],
            [1., -4., 1.],
            [0., 1., 0.]
        ], dtype=torch.float32)

        # Create the target kxk kernel, initialized to zeros
        kernel_kxk = torch.zeros((k, k), dtype=torch.float32)

        # Place the 3x3 kernel in the center of the kxk kernel
        center = k // 2
        start = center - 1
        end = center + 2 # Slice end index is exclusive
        if k >= 3:
            kernel_kxk[start:end, start:end] = laplacian_kernel_3x3
        else: # k=1, kernel is just [[1.]] ? Laplacian needs neighbors. Let's assume k>=3
             kernel_kxk[center, center] = 1.0 # Or handle k=1 case differently if needed

        # Expand kernel to depthwise shape: (channels, 1, k, k)
        depthwise_kernel = kernel_kxk.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)

        # Register as buffer (non-trainable)
        self.register_buffer('laplacian_weight', depthwise_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.laplacian_weight, bias=None, 
                       padding=self.padding, groups=self.channels)

# Main AFB-PE Module
class AFBPE(nn.Module):
    """
    Adaptive Fractional Biharmonic Position Encoding (AFB-PE).

    Combines a learnable fractional Laplacian response (u1) and an approximate
    biharmonic response (u2, via cascaded Laplacians) to generate an attention
    map that modulates the input features, enhancing positional information
    for small targets.

    Args:
        channels (int): Number of input channels.
        r (int): Channel reduction ratio. Defaults to 4.
        k (int): Kernel size for depthwise convolutions. Must be odd. Defaults to 5.
    """
    def __init__(self, channels: int, r: int = 4, k: int = 5):
        super().__init__()
        if k % 2 == 0:
            raise ValueError(f"Kernel size k must be odd, but got {k}")

        self.channels = channels
        self.k = k
        cr = max(8, channels // r) # Reduced channels, ensure minimum
        self.cr = cr

        # --- Channel Reduction ---
        self.reduce = nn.Conv2d(channels, cr, kernel_size=1, bias=False)

        # --- Learnable Fractional Laplacian Branch (u1) ---
        # Learnable alpha parameter
        self.alpha = nn.Parameter(torch.tensor(0.8)) # Initial value suggestion
        # Helper to compute kernel weights based on alpha
        self.frac_kernel_computer = FractionalLaplacianKernel(k)
        # Depthwise conv layer (weights will be set dynamically)
        self.dw_frac = nn.Conv2d(cr, cr, kernel_size=k, padding=k // 2,
                                 groups=cr, bias=False)

        # --- Fixed Cascaded Laplacian Branch (u2 ≈ Biharmonic ∇^4) ---
        self.dw_lap1 = DepthwiseLaplacian(cr, k)
        self.dw_lap2 = DepthwiseLaplacian(cr, k) # Apply Laplacian twice

        # --- Fusion Block ---
        # Fuses u1 and u2 (concatenated) back to original channel dimension
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * cr, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU() # Swish activation
        )

        # --- Final Activation for Modulation ---
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AFB-PE module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor with modulated features, shape (B, C, H, W).
        """
        # --- Input Check ---
        B, C, H, W = x.shape
        if C != self.channels:
             raise ValueError(f"Input channels {C} mismatch module channels {self.channels}")

        # 数值稳定性检查
        if not torch.isfinite(x).all():
            print("Warning: Non-finite input detected in AFBPE")
            x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))

        # --- Channel Reduction ---
        y = self.reduce(x) # Shape: (B, cr, H, W)
        
        # 检查channel reduction后的数值
        if not torch.isfinite(y).all():
            print("Warning: Non-finite values after channel reduction in AFBPE")
            y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))

        # --- Fractional Laplacian Response (u1) ---
        # Compute kernel weights based on current alpha
        # Note: Computation happens on the device where alpha resides
        try:
            frac_kernel_weights = self.frac_kernel_computer(self.alpha) # Shape: (1, 1, k, k)
            
            # 检查kernel weights的有效性
            if not torch.isfinite(frac_kernel_weights).all():
                print("Warning: Non-finite kernel weights in AFBPE, using fallback")
                # 使用简单的Laplacian kernel作为fallback
                fallback_kernel = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]], 
                                             device=frac_kernel_weights.device, 
                                             dtype=frac_kernel_weights.dtype).view(1, 1, 3, 3)
                if self.k > 3:
                    # 如果kernel size大于3，需要padding
                    pad_size = (self.k - 3) // 2
                    fallback_kernel = F.pad(fallback_kernel, [pad_size]*4)
                frac_kernel_weights = fallback_kernel
            
            # Set weights dynamically for the depthwise convolution layer
            # Repeat weights for each input channel (group)
            self.dw_frac.weight.data = frac_kernel_weights.repeat(self.cr, 1, 1, 1)
            # Apply depthwise fractional laplacian
            u1 = self.dw_frac(y) # Shape: (B, cr, H, W)
            
        except Exception as e:
            print(f"Error in fractional Laplacian computation: {e}")
            # 使用identity mapping作为fallback
            u1 = y

        # 检查u1的数值稳定性
        if not torch.isfinite(u1).all():
            print("Warning: Non-finite u1 in AFBPE")
            u1 = torch.where(torch.isfinite(u1), u1, torch.zeros_like(u1))

        # --- Biharmonic Response (u2) ---
        # Apply cascaded fixed depthwise Laplacians
        try:
            u2 = self.dw_lap2(self.dw_lap1(y)) # Shape: (B, cr, H, W)
        except Exception as e:
            print(f"Error in biharmonic computation: {e}")
            u2 = y
            
        # 检查u2的数值稳定性  
        if not torch.isfinite(u2).all():
            print("Warning: Non-finite u2 in AFBPE")
            u2 = torch.where(torch.isfinite(u2), u2, torch.zeros_like(u2))

        # --- Fusion ---
        # Concatenate along the channel dimension
        fused_response = torch.cat([u1, u2], dim=1) # Shape: (B, 2*cr, H, W)
        
        # 检查fusion前的数值
        if not torch.isfinite(fused_response).all():
            print("Warning: Non-finite fused_response in AFBPE")
            fused_response = torch.where(torch.isfinite(fused_response), fused_response, torch.zeros_like(fused_response))
            
        # Apply fusion block (1x1 Conv + BN + SiLU)
        a = self.fuse(fused_response) # Shape: (B, channels, H, W)
        
        # 检查attention map
        if not torch.isfinite(a).all():
            print("Warning: Non-finite attention map in AFBPE")
            a = torch.where(torch.isfinite(a), a, torch.zeros_like(a))

        # --- Residual Modulation ---
        # Calculate modulation factor (attention map) with more conservative scaling
        modulation_factor = 1.0 + 0.1 * self.act(a)  # 减小调制强度防止数值不稳定
        
        # 最终检查
        if not torch.isfinite(modulation_factor).all():
            print("Warning: Non-finite modulation_factor in AFBPE")
            modulation_factor = torch.ones_like(modulation_factor)
            
        # Apply modulation to the original input feature map x
        output = x * modulation_factor # Element-wise multiplication
        
        # 最终输出检查
        if not torch.isfinite(output).all():
            print("Warning: Non-finite output in AFBPE, returning input")
            output = x

        return output

    def get_learnable_params(self) -> Tuple[int, int]:
        """Calculates the number of learnable parameters."""
        params_reduce = sum(p.numel() for p in self.reduce.parameters() if p.requires_grad)
        params_fuse_conv = sum(p.numel() for p in self.fuse[0].parameters() if p.requires_grad)
        params_fuse_bn = sum(p.numel() for p in self.fuse[1].parameters() if p.requires_grad)
        params_alpha = self.alpha.numel()
        # dw_frac, dw_lap1, dw_lap2 weights are either dynamic or fixed (non-learnable)
        total_params = params_reduce + params_fuse_conv + params_fuse_bn + params_alpha
        # Parameters roughly = C*cr (reduce) + (2*cr)*C (fuse conv) + 2*C (fuse bn) + 1 (alpha)
        # Approx: C*cr + 2*C*cr + 2*C + 1 = 3*C*cr + 2*C + 1
        # Since cr = C/r, Approx: 3*C*(C/r) + 2*C + 1 = 3*C^2/r + 2*C + 1
        # The text description says "≈ 1 × 输入通道参数量", which seems incorrect based on this.
        # Let's re-read: "仅多一个可学习标量 α + 两组 1 × 1 Conv 权重。" This aligns better.
        # Params: reduce (C*cr) + fuse[0] (2*cr*C) + fuse[1] (2*C from BN) + alpha (1)
        # Total ≈ C*(C/r) + 2*(C/r)*C + 2*C + 1 = 3*C^2/r + 2*C + 1
        # Maybe the description meant parameter *increase* compared to baseline?
        # Let's just count them accurately.
        num_alpha = params_alpha
        num_conv_weights = params_reduce + params_fuse_conv
        num_bn_weights = params_fuse_bn
        return total_params, num_alpha, num_conv_weights, num_bn_weights


# --- Example Usage ---
if __name__ == '__main__':
    # Example parameters
    batch_size = 2
    input_channels = 64
    height, width = 32, 32
    reduction_ratio = 4
    kernel_size = 5 # Must be odd

    # Create random input tensor
    input_tensor = torch.randn(batch_size, input_channels, height, width)

    # Instantiate the AFB-PE module
    afb_pe_module = AFBPE(channels=input_channels, r=reduction_ratio, k=kernel_size)

    # Test forward pass
    print(f"Input shape: {input_tensor.shape}")
    output_tensor = afb_pe_module(input_tensor)
    print(f"Output shape: {output_tensor.shape}")

    # Check if output shape matches input shape
    assert output_tensor.shape == input_tensor.shape, "Output shape mismatch!"

    # Print number of parameters
    total_params, num_alpha, num_conv, num_bn = afb_pe_module.get_learnable_params()
    print(f"\nAFB-PE Module Parameters (k={kernel_size}, r={reduction_ratio}):")
    print(f"  - Learnable alpha: {num_alpha}")
    print(f"  - Conv weights (1x1): {num_conv}")
    print(f"  - BatchNorm weights: {num_bn}")
    print(f"  - Total Learnable Parameters: {total_params}")

    # Try backward pass to check gradient flow
    try:
        output_tensor.sum().backward()
        print("\nBackward pass successful.")
        # Check if alpha has a gradient
        if afb_pe_module.alpha.grad is not None:
            print("  - Alpha parameter received gradient.")
        else:
            print("  - WARNING: Alpha parameter did NOT receive gradient.")
        # Check gradients for other layers if needed
        # print(f"  - Gradient for reduce layer weight: {afb_pe_module.reduce.weight.grad is not None}")
        # print(f"  - Gradient for fuse layer [0] weight: {afb_pe_module.fuse[0].weight.grad is not None}")
        # print(f"  - Gradient for fuse layer [1] weight: {afb_pe_module.fuse[1].weight.grad is not None}")

    except Exception as e:
        print(f"\nError during backward pass: {e}")

    # Example with different kernel size
    print("-" * 30)
    kernel_size_3 = 3
    afb_pe_module_k3 = AFBPE(channels=input_channels, r=reduction_ratio, k=kernel_size_3)
    input_tensor_k3 = torch.randn(batch_size, input_channels, height, width)
    output_tensor_k3 = afb_pe_module_k3(input_tensor_k3)
    total_params_k3, _, _, _ = afb_pe_module_k3.get_learnable_params()
    print(f"AFB-PE Module (k={kernel_size_3}) Output shape: {output_tensor_k3.shape}")
    print(f"AFB-PE Module (k={kernel_size_3}) Total Learnable Parameters: {total_params_k3}")

# ------------- New AFBPEConv Module -------------
from ultralytics.nn.modules.conv import Conv

def autopad(k, p=None, d=1):
    """Auto-padding calculation"""
    if p is None:
        p = (k - 1) // 2 * d
    return p

class AFBPEConv(nn.Module):
    """
    Convolutional layer combined with Adaptive Fractional Biharmonic Position Encoding.

    Applies a standard convolution (Conv), followed by the AFBPE module.
    The AFBPE module enhances feature representations by encoding fractional biharmonic
    information, particularly useful for small target detection.

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
        afbpe_r (int): Channel reduction ratio for AFBPE. Defaults to 4.
        afbpe_k (int): Kernel size for depthwise convolutions in AFBPE. Must be odd. Defaults to 5.
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, 
                 afbpe_r=4, afbpe_k=5):
        super().__init__()
        # Standard convolution layer
        self.conv = Conv(c1, c2, k, s, p=p, g=g, d=d, act=act)
        # AFBPE module applied on the output of the convolution (c2 channels)
        self.afbpe = AFBPE(channels=c2, r=afbpe_r, k=afbpe_k)

    def forward(self, x):
        """Applies Conv -> AFBPE to the input tensor."""
        y = self.conv(x)
        return self.afbpe(y)

    def forward_fuse(self, x):
        """
        Forward pass for fused model. 
        Applies the fused Conv and then the AFBPE module.
        """
        y = self.conv.forward_fuse(x)
        return self.afbpe(y)
