import torch
import torch.nn as nn


class Cls_Boost_head(nn.Module):
    def __init__(self, in_ch, num_classes, ch_reduce=16):
        super().__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_c = nn.AdaptiveAvgPool2d((1))
        self.conv_1x1 = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(in_ch, num_classes)
        self.ch_attn = nn.Sequential(nn.Linear(in_ch, in_ch // ch_reduce),
                                     nn.GELU(),
                                     nn.Linear(in_ch // ch_reduce, in_ch),
                                     nn.Sigmoid())
        self.linear = nn.Linear(in_ch, num_classes)

    def forward(self, x, H, W):

        B, L, C = x.shape  # Batch size, H*W, Channel

        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, L, C] -> [B, C, H, W]

        # 空间注意力
        x_h = self.pool_h(x)  # [B, C, H, W] -> [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, H, W] -> [B, C, W, 1]
        hw = self.conv_1x1(torch.cat([x_h, x_w], dim=2))  # [B, C, H+W, 1]
        x_h, x_w = torch.split(hw, [H, W], dim=2)  # [B, C， H, 1], [B, C, W, 1]
        x_h = x_h.sigmoid()   # [B, C， H, 1]
        x_w = x_w.permute(0, 1, 3, 2).sigmoid()  # [B, C, W, 1] -> [B, C, 1, W]
        x = x * x_h * x_w  # [B, C， H, W] * [B, C， H, 1] * [B, C, 1, W]

        # 通道注意力
        x = self.avg_pool(x)  # [B, C, H, W] -> [B, C, 1, 1]
        x = torch.flatten(x, 1)  # [B, C, 1, 1] -> [B, C]
        y = self.ch_attn(x)
        x = x * y
        x = self.linear(x)

        return x






















