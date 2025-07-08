import torch
import torch.nn as nn
import torch.nn.functional as F

class ADRC_PE(nn.Module):
    """
    Adaptive Discrete-Ricci Curvature Position Encoding (ADRC-PE).

    Uses an approximation of discrete Ollivier-Ricci curvature on the pixel grid
    to distinguish between peaks (potential small targets, associated with positive
    curvature) and textures/edges (negative/zero curvature). The curvature map
    is gated by an adaptive mechanism based on global context and fused into an
    attention map applied to the input features.

    Implemented primarily with fixed-kernel depthwise convolutions and a few
    learnable 1x1 convolutions for efficiency.

    Args:
        c (int): Number of input channels.
        r (int): Channel reduction ratio for the initial 1x1 convolution.
                 Intermediate channels will be max(8, c // r). Defaults to 4.
        eps (float): Small epsilon value to prevent division by zero when
                     calculating curvature. Defaults to 1e-4.
    """
    def __init__(self, c: int, r: int = 4, eps: float = 1e-4):
        super().__init__()
        if r <= 0:
            raise ValueError("Reduction ratio `r` must be positive.")

        # Calculate reduced channels, ensuring a minimum (e.g., 8)
        c_red = max(8, c // r)
        self.cr = c_red  # Store for use in forward method
        self.eps = eps

        # 1x1 Convolution for channel reduction
        self.reduce = nn.Conv2d(c, c_red, kernel_size=1, stride=1, padding=0, bias=False)
        # Add normalization layer for stability
        self.norm = nn.GroupNorm(num_groups=min(8, c_red), num_channels=c_red)

        # Fixed 3x3 averaging kernel for local mean calculation
        mean_kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32) / 9.0
        # Repeat kernel for all `c_red` channels (depthwise)
        mean_kernel_depthwise = mean_kernel.repeat(c_red, 1, 1, 1)
        # Register as buffer (non-trainable)
        self.register_buffer('mean3_weight', mean_kernel_depthwise)

        # Fixed Sobel filters for calculating gradient magnitude |∇X|
        # Sobel X kernel
        sobel_x_kernel = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0 # Normalization factor
        sobel_x_kernel_depthwise = sobel_x_kernel.repeat(c_red, 1, 1, 1)
        # Register as buffer (non-trainable)
        self.register_buffer('gradx_weight', sobel_x_kernel_depthwise)

        # Sobel Y kernel (transpose of Sobel X)
        sobel_y_kernel = sobel_x_kernel.transpose(-1, -2) # Transpose H and W dimensions
        sobel_y_kernel_depthwise = sobel_y_kernel.repeat(c_red, 1, 1, 1)
        # Register as buffer (non-trainable)
        self.register_buffer('grady_weight', sobel_y_kernel_depthwise)

        # Adaptive gating mechanism (MLP on global context)
        gate_channels_hidden = max(1, c_red // 4) # Ensure at least 1 channel
        self.mlpgate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_red, gate_channels_hidden, kernel_size=1, bias=True), # Can use bias here
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_channels_hidden, c_red, kernel_size=1, bias=True), # Output c_red channels to match kappa
            nn.Sigmoid() # Output gate values between 0 and 1
        )

        # Fusion layer to combine curvature information into a single attention map
        # Input channels = 2 * c_red (concatenated kappa and gated kappa)
        # Output channels = c (to match original input x for residual connection)
        # Note: The description snippet showed Conv2d(2, 1, ...), which seemed inconsistent
        # with kappa having c_red channels. Assuming the concatenation yields 2*c_red channels
        # and the final attention map should modulate the original 'c' channels.
        # Let's output 'c' channels directly for the modulation.
        # Alternative: output 1 channel and broadcast, as in PHDE. Let's follow PHDE structure.
        # Output 1 channel attention map. Adjusting Conv2d input channels.
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * c_red, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid() # Output attention map in range [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ADRC_PE module.

        Args:
            x (torch.Tensor): Input feature map (B x C x H x W).

        Returns:
            torch.Tensor: Output feature map with positional encoding applied
                          (B x C x H x W).
        """
        b, c, h, w = x.shape

        # 1. Reduce channels and normalize
        y = self.reduce(x) # B x c_red x H x W
        y = self.norm(y)   # Apply normalization for stability

        # 2. Calculate local mean using fixed kernel
        mu = F.conv2d(y, self.mean3_weight, bias=None, stride=1, padding=1, groups=self.cr) # B x c_red x H x W

        # 3. Calculate gradient magnitude using fixed Sobel filters
        grad_x = F.conv2d(y, self.gradx_weight, bias=None, stride=1, padding=1, groups=self.cr)
        grad_y = F.conv2d(y, self.grady_weight, bias=None, stride=1, padding=1, groups=self.cr)
        grad_mag = torch.abs(grad_x) + torch.abs(grad_y) # B x c_red x H x W

        # 4. Estimate discrete Ricci curvature (simplified approximation)
        # kappa = 1.0 - |y - mu| / (|∇y| + ε)
        # Apply normalization to prevent extreme values
        diff_norm = torch.abs(y - mu)
        grad_norm = grad_mag + self.eps
        ratio = diff_norm / grad_norm
        # Clamp ratio to prevent extreme curvature values
        ratio = torch.clamp(ratio, 0.0, 2.0)
        kappa = 1.0 - ratio # B x c_red x H x W
        # Clamp final curvature to reasonable range
        kappa = torch.clamp(kappa, -1.0, 1.0)

        # 5. Calculate adaptive gate gamma (γ)
        gamma = self.mlpgate(y) # B x c_red x 1 x 1 (spatial dimensions are squeezed by pool)
                                # This needs broadcasting to HxW for element-wise mult

        # 6. Fuse curvature and gated curvature
        # Concatenate kappa and gamma*kappa along the channel dimension
        # gamma needs to match spatial dims of kappa for element-wise product
        # gamma shape is B x c_red x 1 x 1, kappa shape is B x c_red x H x W
        # Broadcasting applies gamma element-wise across H, W.
        gated_kappa = gamma * kappa # B x c_red x H x W
        fused_input = torch.cat([kappa, gated_kappa], dim=1) # B x (2*c_red) x H x W

        # 7. Generate attention map 'a'
        a = self.fuse(fused_input) # B x 1 x H x W (Attention map)
        
        # 8. Apply attention map to original input using residual connection
        # Use a more gentle modulation to prevent extreme feature changes
        # Scale down the attention effect to 0.1 times for stability
        a = a * 0.1  # Reduce attention strength
        return x * (1.0 + a)

# Example Usage:
if __name__ == '__main__':
    # Example parameters
    input_channels = 64
    batch_size = 4
    height, width = 32, 32

    # Instantiate the module
    # Use default r=4, eps=1e-4
    adrc_pe_module = ADRC_PE(c=input_channels)
    # print(adrc_pe_module)

    # Create a dummy input tensor
    dummy_input = torch.randn(batch_size, input_channels, height, width)

    # Forward pass
    output = adrc_pe_module(dummy_input)

    # Check output shape
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Should be same as input

    # Calculate number of parameters (optional)
    num_params_total = sum(p.numel() for p in adrc_pe_module.parameters())
    num_params_trainable = sum(p.numel() for p in adrc_pe_module.parameters() if p.requires_grad)
    print(f"Number of total parameters in ADRC_PE: {num_params_total}")
    print(f"Number of trainable parameters in ADRC_PE: {num_params_trainable}")

    # Verify some internal shapes (optional)
    adrc_pe_module.eval()
    with torch.no_grad():
        y = adrc_pe_module.reduce(dummy_input)
        mu = adrc_pe_module.mean3(y)
        gx = adrc_pe_module.gradx(y)
        gy = adrc_pe_module.grady(y)
        grad = torch.abs(gx) + torch.abs(gy)
        kappa = 1.0 - torch.abs(y - mu) / (grad + adrc_pe_module.eps)
        gamma = adrc_pe_module.mlpgate(y)
        fused_input = torch.cat([kappa, gamma * kappa], dim=1)
        a = adrc_pe_module.fuse(fused_input)

        print(f"Shape of reduced 'y': {y.shape}") # B x c_red x H x W
        print(f"Shape of 'kappa': {kappa.shape}") # B x c_red x H x W
        print(f"Shape of 'gamma': {gamma.shape}") # B x c_red x 1 x 1
        print(f"Shape of 'fused_input': {fused_input.shape}") # B x 2*c_red x H x W
        print(f"Shape of attention 'a': {a.shape}") # B x 1 x H x W

# ------------- New ADRCConv Module -------------
from ultralytics.nn.modules.conv import Conv

class ADRCConv(nn.Module):
    """
    Convolutional layer combined with Adaptive Discrete-Ricci Curvature Position Encoding.

    Applies a standard convolution (Conv), followed by the ADRC_PE module.
    The ADRC_PE is applied to the output features of the Conv layer.

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
        adrc_r (int): Channel reduction ratio for ADRC_PE. Defaults to 4.
        adrc_eps (float): Epsilon value for ADRC_PE. Defaults to 1e-4.
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, adrc_r=8, adrc_eps=1e-2):
        super().__init__()
        # Standard convolution layer
        self.conv = Conv(c1, c2, k, s, p=p, g=g, d=d, act=act)
        # ADRC Position Encoding applied on the output of the convolution (c2 channels)
        self.adrc_pe = ADRC_PE(c=c2, r=adrc_r, eps=adrc_eps)

    def forward(self, x):
        """Applies Conv -> ADRC_PE to the input tensor."""
        y = self.conv(x)
        return self.adrc_pe(y)

    def forward_fuse(self, x):
        """
        Forward pass for fused model. Assumes ADRC_PE might not be fusable
        or fusion behavior is just applying the fused Conv followed by ADRC_PE.
        """
        y = self.conv.forward_fuse(x)
        return self.adrc_pe(y)

    # Note: fuse_convs method is specific to RepConv or similar architectures
    # that have parallel branches to fuse. ADRCConv doesn't have parallel
    # Conv branches in its definition, only a sequential Conv -> ADRC_PE.
    # The internal self.conv might be fusable if it's a type like Conv2 or RepConv,
    # but fusing ADRC_PE itself isn't straightforward.
    # If fusion is needed, the self.conv.fuse_convs() could potentially be called.
