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
        self.eps = eps

        # 1x1 Convolution for channel reduction
        self.reduce = nn.Conv2d(c, c_red, kernel_size=1, stride=1, padding=0, bias=False)

        # Depthwise convolution for calculating local mean (μ) with fixed kernel
        self.mean3 = nn.Conv2d(
            c_red, c_red, kernel_size=3, stride=1, padding=1, # padding=1 preserves size
            groups=c_red, bias=False
        )
        # Define 3x3 averaging kernel
        mean_kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32) / 9.0
        # Repeat kernel for all `c_red` channels (depthwise)
        mean_kernel_depthwise = mean_kernel.repeat(c_red, 1, 1, 1)
        # Assign fixed kernel and make it non-trainable
        self.mean3.weight.data.copy_(mean_kernel_depthwise)
        self.mean3.weight.requires_grad_(False)

        # Depthwise convolutions for calculating gradient magnitude |∇X| using Sobel filters
        # Sobel X kernel
        sobel_x_kernel = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0 # Normalization factor
        sobel_x_kernel_depthwise = sobel_x_kernel.repeat(c_red, 1, 1, 1)

        self.gradx = nn.Conv2d(c_red, c_red, 3, 1, 1, groups=c_red, bias=False)
        self.gradx.weight.data.copy_(sobel_x_kernel_depthwise)
        self.gradx.weight.requires_grad_(False)

        # Sobel Y kernel (transpose of Sobel X)
        sobel_y_kernel = sobel_x_kernel.transpose(-1, -2) # Transpose H and W dimensions
        sobel_y_kernel_depthwise = sobel_y_kernel.repeat(c_red, 1, 1, 1)

        self.grady = nn.Conv2d(c_red, c_red, 3, 1, 1, groups=c_red, bias=False)
        self.grady.weight.data.copy_(sobel_y_kernel_depthwise)
        self.grady.weight.requires_grad_(False)

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

        # 1. Reduce channels
        y = self.reduce(x) # B x c_red x H x W

        # 2. Calculate local mean
        mu = self.mean3(y) # B x c_red x H x W

        # 3. Calculate gradient magnitude (using L1 norm approximation)
        grad_x = self.gradx(y)
        grad_y = self.grady(y)
        grad_mag = torch.abs(grad_x) + torch.abs(grad_y) # B x c_red x H x W

        # 4. Estimate discrete Ricci curvature (simplified approximation)
        # kappa = 1.0 - |y - mu| / (|∇y| + ε)
        kappa = 1.0 - torch.abs(y - mu) / (grad_mag + self.eps) # B x c_red x H x W
        # Optional: Clamp kappa to a reasonable range, e.g., [-1, 1] or [0, 1] if desired,
        # though the formula doesn't strictly guarantee this range.
        # kappa = torch.clamp(kappa, -1.0, 1.0)

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
        # x * (1 + a) -> broadcasts 'a' across the original C channels
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