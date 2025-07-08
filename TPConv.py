import torch
import torch.nn as nn
import torch.nn.functional as F

"""
TPConv – baseline (<40 lines)。多激活/均值-方差 proxy。
Depth-Wise  基座提取局部场。
`num_pathways` 条 *1×1* probe 分支 → 不同 **强度区间** 感知（通过激活或 bias）。
无 BatchNorm，保证响应幅值物理可解释。

"""
#  TPConv – mean/std 持久性代理
class TPConv(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        *,
        num_pathways: int = 4,
        reduction: int = 1,
        use_bias_shift: bool = True,
    ) -> None:
        """Baseline Topological-Persistence Convolution.

        Args:
            c1, c2: in/out channels.
            k, s: depth-wise conv kernel & stride.
            num_pathways: parallel intensity probes.
            reduction: channel reduction ratio inside probe.
            use_bias_shift: if True, all probes共享 ReLU + learnable bias.
        """
        super().__init__()
        self.use_bias_shift = use_bias_shift

        # 1. Depth-wise base
        self.base = nn.Sequential(
            nn.Conv2d(c1, c1, k, s, padding=k // 2, groups=c1, bias=False),
            nn.SiLU(),
        )

        # 2. Build probes
        mid = max(c1 // reduction, 8)
        self.num_pathways = num_pathways
        self.pathways = nn.ModuleList()
        if use_bias_shift:
            # 单一 ReLU 激活 + 可学习 bias
            self.bias = nn.Parameter(torch.zeros(num_pathways, 1, 1, 1))
            for _ in range(num_pathways):
                self.pathways.append(
                    nn.Sequential(
                        nn.Conv2d(c1, mid, 1, bias=False),
                        nn.ReLU(),
                        nn.Conv2d(mid, c1, 1, bias=False),
                    )
                )
        else:
            # 多样激活函数覆盖不同区段
            acts: list[nn.Module] = [
                nn.Hardtanh(0.6, 1.0),  # narrow high-intensity window
                nn.ReLU(),              # broad positive
                nn.Sigmoid(),           # global smooth 0-1
                nn.Tanh(),              # symmetric saturating
                nn.Hardswish(),         # piecewise smooth
                nn.GELU(),              # Gaussian-like
            ]
            for i in range(num_pathways):
                self.pathways.append(
                    nn.Sequential(
                        nn.Conv2d(c1, mid, 1, bias=False),
                        acts[i % len(acts)],
                        nn.Conv2d(mid, c1, 1, bias=False),
                    )
                )

        # 3. Fuse mean + std
        self.fuse = nn.Conv2d(c1 * 2, c1, 1, bias=False)
        # 4. Projection
        self.project = nn.Conv2d(c1, c2, 1, bias=False)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, *, return_maps: bool = False):
        f_base = self.base(x)
        probes = []
        for idx, p in enumerate(self.pathways):
            if self.use_bias_shift:
                probes.append(p(f_base + self.bias[idx]))
            else:
                probes.append(p(f_base))
        stack = torch.stack(probes)  # (P,B,C,H,W)
        # implicit persistence proxy
        y = self.fuse(torch.cat([stack.mean(0), stack.std(0)], dim=1))
        y = self.project(y)
        if return_maps:
            return y, {
                "base": f_base.detach(),
                "probes": [t.detach() for t in probes],
                "persistence": y.detach(),
            }
        return y


