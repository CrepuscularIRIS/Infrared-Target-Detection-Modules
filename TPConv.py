import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Plug-and-Play **Topological-Persistence Convolution** family
-----------------------------------------------------------
Three drop-in modules, sharing相同  API：

* **TPConv** – baseline (<40 lines)。多激活/均值-方差 proxy。
* **TPConvAttn** – 轻量注意力版本，用 softmax 权重替代 mean/std。
* **TPConvMS** – 多空间尺度 + intensity 的持久性融合。

公共特性
~~~~~~~~
* *Depth-Wise* 基座提取局部场。
* `num_pathways` 条 *1×1* probe 分支 → 不同 **强度区间** 感知（通过激活或 bias）。
* `return_maps=True` 时，返回适合论文热力图的中间输出（base, branch, persistence）。
* **无 BatchNorm**，保证响应幅值物理可解释。

Usage
~~~~~
```python
x = torch.randn(2, 64, 128, 128)
mod = TPConv(64, 64)
y, maps = mod(x, return_maps=True)
```

Notes
~~~~~
* 所有模块均在 PyTorch>=1.10 上零额外依赖。
* 为保持极简，attention 实现基于逐像素 softmax 权重；若需更强力可替换为 MHSA。
"""

# -----------------------------------------------------------------------------
#  1. Baseline TPConv – mean/std 持久性代理
# -----------------------------------------------------------------------------

class TPConv(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        *,
        num_pathways: int = 6,
        reduction: int = 8,
        use_bias_shift: bool = False,
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

# -----------------------------------------------------------------------------
#  2. TPConvAttn – learnable路径权重（softmax attention）
# -----------------------------------------------------------------------------

class TPConvAttn(TPConv):
    """Attention variant: learnable pathway weights instead of mean/std."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # replace fuse layer with softmax attention
        self.weight_mlp = nn.Conv2d(self.pathways[0][0].out_channels, 1, 1)
        # output conv重用父类 self.project

    def forward(self, x: torch.Tensor, *, return_maps: bool = False):
        f_base = self.base(x)
        probes = [p(f_base) for p in self.pathways]  # list[(B,C,H,W)]
        stack = torch.stack(probes, dim=0)           # (P,B,C,H,W)

        # compute per-pathway importance (softmax over P)
        attn_logits = self.weight_mlp(stack)         # (P,B,1,H,W)
        attn = F.softmax(attn_logits, dim=0)
        fused = (attn * stack).sum(dim=0)            # (B,C,H,W)
        y = self.project(fused)
        if return_maps:
            return y, {
                "base": f_base.detach(),
                "probes": [t.detach() for t in probes],
                "attn": attn.detach(),
                "persistence": fused.detach(),
            }
        return y

# -----------------------------------------------------------------------------
#  3. TPConvMS – 多空间尺度 + intensity 持久性
# -----------------------------------------------------------------------------

class _DW(nn.Module):
    def __init__(self, c, k, dilation=1):
        super().__init__()
        pad = dilation * (k // 2)
        self.conv = nn.Conv2d(c, c, k, padding=pad, groups=c, bias=False, dilation=dilation)
    def forward(self, x):
        return self.conv(x)

class TPConvMS(TPConv):
    """Multi-Scale variant: each pathway uses different spatial kernel."""
    def __init__(self, c1, c2, *, scales=(3, 5, 7), **kwargs):
        num_pathways = len(scales)
        super().__init__(c1, c2, num_pathways=num_pathways, **kwargs)
        # override pathway list with DWConv(k) + 1×1
        self.pathways = nn.ModuleList()
        mid = max(c1 // kwargs.get("reduction", 8), 8)
        for k in scales:
            self.pathways.append(
                nn.Sequential(
                    _DW(c1, k),
                    nn.SiLU(),
                    nn.Conv2d(c1, mid, 1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(mid, c1, 1, bias=False),
                )
            )
        # reuse parent's fuse/project

# -----------------------------------------------------------------------------
#  4. Ablation Variants
# -----------------------------------------------------------------------------
class TPConvBaseOnly(nn.Module):
    """仅 base conv + projection；无任何路径。"""
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.base = nn.Sequential(_DW(c1, k, s), nn.SiLU())
        self.project = nn.Conv2d(c1, c2, 1, bias=False)
    def forward(self, x, *, return_maps=False):
        y = self.project(self.base(x))
        return (y, {}) if return_maps else y

class TPConvMeanOnly(TPConv):
    """fusion 只用均值，无 std。"""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.fuse_mean = nn.Conv2d(a[0] if a else kw['c1'] * 1, a[0] if a else kw['c1'], 1, bias=False)
    def forward(self, x, *, return_maps=False):
        f_base = self.base(x)
        stack = self._probe_stack(f_base)
        y = self.fuse_mean(stack.mean(0))
        y = self.project(y)
        if return_maps:
            return y, {"base": f_base.detach(), "probes": stack.detach(), "persistence": y.detach()}
        return y

class TPConvStdOnly(TPConv):
    """fusion 只用 std，无均值。"""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.fuse_std = nn.Conv2d(a[0] if a else kw['c1'] * 1, a[0] if a else kw['c1'], 1, bias=False)
    def forward(self, x, *, return_maps=False):
        f_base = self.base(x)
        stack = self._probe_stack(f_base)
        y = self.fuse_std(stack.std(0))
        y = self.project(y)
        if return_maps:
            return y, {"base": f_base.detach(), "probes": stack.detach(), "persistence": y.detach()}
        return y

class TPConvUniformAct(TPConv):
    """所有 probe 用相同 ReLU，考察激活多样性作用。"""
    def __init__(self, c1, c2, **kw):
        super().__init__(c1, c2, **kw)
        mid = max(c1 // kw.get("reduction", 8), 8)
        self.pathways = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c1, mid, 1, bias=False), nn.ReLU(),
                nn.Conv2d(mid, c1, 1, bias=False)) for _ in range(self.pathways.__len__())
        ])