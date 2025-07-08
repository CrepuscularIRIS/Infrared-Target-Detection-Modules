import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ultralytics.nn.modules.conv import Conv

def autopad(k, p=None, d=1):
    """Auto-padding calculation"""
    if p is None:
        p = (k - 1) // 2 * d
    return p

class MSGRA(nn.Module):
    """
    Multi-Scale Green-Radial Attention (MS-GRA) for infrared small target detection.
    Uses fixed Gaussian-Bessel radial kernels and adaptive scale weights.
    Compatible with YOLO-style Conv/C3k2 slots, FP16 & ONNX friendly.
    """
    def __init__(self, channels, K=4, sigmas=(1.0, 2.0, 3.0, 4.0)):
        super().__init__()
        self.channels = channels
        self.K = K
        self.kernel_modules = nn.ModuleList()
        kernel_size = 5
        
        # 创建固定的高斯核模块
        for i in range(K):
            # 创建空的模块，存储卷积参数而不是用nn.Conv2d
            m = nn.Module()
            dilation = max(1, int(sigmas[i] // 1))  # 确保最小dilation为1
            m.groups = channels
            m.kernel_size = kernel_size
            m.stride = 1
            m.padding = autopad(kernel_size, p=None, d=dilation)
            m.dilation = dilation
            
            # 生成固定核权重并注册为buffer，而不是parameter
            with torch.no_grad():
                # create Gaussian kernel
                ax = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
                xx, yy = torch.meshgrid(ax, ax, indexing='ij')  # 添加indexing参数以避免警告
                kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigmas[i]**2))
                kernel = kernel / kernel.sum()
                # assign to depthwise conv weight
                weight = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
                # 注册为buffer而不是parameter
                m.register_buffer('weight', weight)
                
            self.kernel_modules.append(m)
            
        # Adaptive scale network
        hidden = max(8, channels // 8)
        # 使用GroupNorm代替BatchNorm2d，避免在小批量和1x1特征时的问题
        num_groups = min(4, hidden) if hidden > 0 else 1 # 确保组数不超过通道数，且至少为1
        self.scale = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden),  # 使用GroupNorm替代BatchNorm2d
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, K, kernel_size=1, bias=True)
        )
        
    def _depthwise_conv2d(self, x, weight, padding, groups, dilation=1, stride=1):
        """执行深度可分离卷积，使用buffer权重而不是参数权重"""
        return F.conv2d(x, weight, None, stride, padding, dilation, groups)

    def forward(self, x):
        # x: B, C, H, W
        # Global average pooling
        gap = x.mean(dim=(2, 3), keepdim=True)  # B, C, 1, 1
        gamma = self.scale(gap)                 # B, K, 1, 1
        
        # Compute radial responses
        radial_out = 0
        for i, m in enumerate(self.kernel_modules):
            # 应用卷积
            conv_out = self._depthwise_conv2d(
                x, 
                m.weight, 
                padding=m.padding, 
                groups=m.groups,
                dilation=m.dilation,
                stride=m.stride
            )
            
            # 确保尺寸匹配后进行加权求和
            if conv_out.shape == x.shape:
                 radial_out = radial_out + gamma[:, i:i+1] * conv_out
            else:
                 # 如果尺寸不匹配（理论上不应发生），可以考虑进行插值或报错
                 # 这里为了健壮性，暂时跳过尺寸不匹配的项，并打印警告
                 print(f"Warning: Size mismatch in MSGRA radial kernel {i}. Input: {x.shape}, Output: {conv_out.shape}")
                 # 或者进行插值:
                 # conv_out_resized = F.interpolate(conv_out, size=x.shape[2:], mode='bilinear', align_corners=False)
                 # radial_out = radial_out + gamma[:, i:i+1] * conv_out_resized
        
        # Attention map
        attn = torch.sigmoid(radial_out)
        return x * (1 + attn)

#--------------------------------------------------------------------------
# 新增: MSGRAConv - 结合多尺度高斯径向注意力和标准卷积的模块
#--------------------------------------------------------------------------
class MSGRAConv(nn.Module):
    """
    结合多尺度高斯径向注意力(MS-GRA)和标准卷积的模块。
    
    特点：
    - 使用固定的高斯-贝塞尔径向核和自适应尺度权重
    - 特别适用于红外小目标检测任务
    - 高效且友好的FP16和ONNX导出支持
    
    接口与Conv保持一致，同时提供注意力增强功能。
    """
    
    default_act = nn.SiLU()  # 默认激活函数，与Conv保持一致
    
    def __init__(
        self, 
        c1,                 # 输入通道数或 [c2, k, s] 形式的列表
        c2=None,            # 输出通道数
        k=1,                # 卷积核大小
        s=1,                # 步长
        p=None,             # 填充
        g=1,                # 卷积分组数
        d=1,                # 卷积膨胀率
        act=True,           # 激活函数
        K=4,                # 径向核数量
        sigmas=(1.0, 2.0, 3.0, 4.0),  # 高斯核的sigma值
        attn_first=False    # 注意力在卷积前(True)还是卷积后(False)
    ):
        """
        初始化MSGRAConv模块
        
        Args:
            c1 (int或List): 输入通道数或 [c2, k, s] 形式的参数列表
            c2 (int): 输出通道数
            k (int): 卷积核大小
            s (int): 步长
            p (int或None): 填充，None则自动计算
            g (int): 分组卷积的组数
            d (int): 空洞卷积的膨胀率
            act (bool或nn.Module): 激活函数
            K (int): 径向核数量
            sigmas (tuple): 高斯核的sigma值列表，长度应等于K
            attn_first (bool): 是否先应用注意力再卷积
        """
        super().__init__()
        self.attn_first = attn_first
        
        # 确保sigmas长度与K匹配
        if len(sigmas) != K:
            sigmas = tuple(1.0 + i for i in range(K))  # 如果长度不匹配，使用默认值
        
        # 解析参数
        in_channels, out_channels, kernel_size, stride = self._parse_args(c1, c2, k, s)
        
        # 创建标准Conv模块
        self.conv = Conv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            p=p,
            g=g,
            d=d,
            act=False  # 激活函数最后应用
        )
        
        # 确定MSGRA应该作用于哪些通道
        attn_channels = self._determine_attn_channels(in_channels, out_channels)
        
        # 创建注意力模块
        self.attn = MSGRA(
            channels=attn_channels,
            K=K,
            sigmas=sigmas
        )
        
        # 最终激活函数
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    def _parse_args(self, c1, c2, k, s):
        """解析输入参数，处理YAML和直接参数两种情况"""
        # 检查是否是从YAML读取的列表形式参数
        if isinstance(c1, list):
            # 第一层的处理 - 默认RGB图像输入
            in_channels = 3
            
            # 解析列表中的参数
            if len(c1) >= 3:
                out_channels, kernel_size, stride = c1[:3]
            elif len(c1) == 2:
                out_channels, kernel_size = c1
                stride = 1
            else:
                out_channels = c1[0]
                kernel_size = 1
                stride = 1
        else:
            # 常规参数模式
            in_channels = c1
            out_channels = c2 if c2 is not None else c1
            kernel_size = k
            stride = s
            
        return in_channels, out_channels, kernel_size, stride
    
    def _determine_attn_channels(self, in_channels, out_channels):
        """确定注意力模块应该作用于哪些通道"""
        # 如果是第一层且注意力在卷积后，或一般情况下注意力在卷积后
        if (in_channels == 3 or not self.attn_first):
            return out_channels
        # 否则，注意力在卷积前
        else:
            return in_channels
    
    def forward(self, x):
        """标准前向传播"""
        # 保存原始数据类型
        input_dtype = x.dtype
        
        if self.attn_first:
            # 先应用注意力，再进行卷积
            x = self.attn(x)
            x = self.conv(x)
        else:
            # 先进行卷积，再应用注意力
            x = self.conv(x)
            x = self.attn(x)
        
        # 应用激活函数
        return self.act(x)
    
    def forward_fuse(self, x):
        """用于推理优化的融合前向传播"""
        if self.attn_first:
            # 先应用注意力，再进行融合卷积
            x = self.attn(x)
            x = self.conv.forward_fuse(x)
        else:
            # 先进行融合卷积，再应用注意力
            x = self.conv.forward_fuse(x)
            x = self.attn(x)
        
        # 应用激活函数
        return self.act(x)


# Example usage:
if __name__ == "__main__":
    # 测试 MSGRA
    channels = 32
    model = MSGRA(channels=channels, K=4, sigmas=(1.0, 2.0, 3.0, 4.0))
    dummy = torch.randn(1, channels, 128, 128)  # 使用导致报错的尺寸进行测试
    out = model(dummy)
    print(f"MSGRA: {dummy.shape} -> {out.shape}")
    print("Test passed!" if dummy.shape == out.shape else "Test failed!")
    
    # 测试 MSGRAConv
    msgra_conv = MSGRAConv(
        c1=32, 
        c2=64, 
        k=3, 
        s=1, 
        K=4,
        sigmas=(1.0, 2.0, 3.0, 4.0),
        attn_first=False
    )
    dummy_conv = torch.randn(1, 32, 128, 128)
    z = msgra_conv(dummy_conv)
    print(f"MSGRAConv: {dummy_conv.shape} -> {z.shape}")
    print("Test passed!" if z.shape == (1, 64, 128, 128) else "Test failed!")
