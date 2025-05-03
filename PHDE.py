import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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