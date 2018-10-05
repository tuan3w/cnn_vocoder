import torch
import torch.nn.functional as F
from torch import nn

class ResnetBlock(nn.Module):
    """Residual Block
    Args:
        in_channels (int): number of channels in input data
        out_channels (int): number of channels in output 
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, one_d=False):
        super(ResnetBlock, self).__init__()
        self.build_conv_block(in_channels, out_channels, one_d, kernel_size=kernel_size)

    def build_conv_block(self, in_channels, out_channels, one_d, kernel_size=3):
        padding = (kernel_size -1)//2
        if not one_d:
            conv = nn.Conv2d
            norm = nn.BatchNorm2d
        else:
            conv = nn.Conv1d
            norm = nn.BatchNorm1d

        self.conv1 = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            norm(out_channels),
            nn.ELU()
        )
        self.conv2 = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            norm(out_channels),
        )
        if in_channels != out_channels:
            self.down = nn.Sequential(
                conv(in_channels, out_channels, kernel_size=1, bias=False),
                norm(out_channels)
            )
        else:
            self.down = None
        
        self.act = nn.ELU()

    def forward(self, x):
        """
        Args:
            x (Tensor): B x C x T
        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.down is not None:
            residual = self.down(residual)
        return self.act(out + residual)

class UpsamplingLayer(nn.Module):
    """Applies 1D upsampling operator over input tensor.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        residuals (int, optional): number of residual blocks. Default=0
    """
    def __init__(self, in_channels, out_channels, residuals=0):
        super(UpsamplingLayer, self).__init__()
        # TODO: try umsampling with bilinear interpolation 
        self.upsample = nn.Upsample(scale_factor=2, mode='linear')
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv.weight)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ELU()

        if residuals != 0:
            # resnet blocks
            layers = []
            for _ in range(residuals):
                layers.append(
                    ResnetBlock(out_channels, out_channels, one_d=True)
                    )
            self.res_blocks = nn.Sequential(*layers)
        else:
            self.res_blocks = None


    def forward(self, x):
        """
        Args:
            x (Tensor): B x in_channels x T
        
        Returns:
            Tensor of shape (B, out_channels, T x 2)
        """
        # upsample network
        B, C, T = x.shape
        # upsample
        # x = x.unsqueeze(dim=3)
        # x = F.upsample(x, size=(T*2, 1), mode='bilinear').squeeze(3)
        x = self.upsample(x)
        # x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        # pass through resnet blocks to improve internal representations
        # of data
        if self.res_blocks != None:
            x = self.res_blocks(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ELU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class Head(nn.Module):
    """Head module

    Args:
        channels (list): list of #channels in each upsampling layer
        pre_residuals (int, optional): number of residual blocks before upsampling. Default: 64
        down_conv_channels (list): list of #channels in each down_conv blocks
        up_residuals (int, optional): number of residual blocks in each upsampling module. Default: 0
    """
    def __init__(self, channels, 
          pre_residuals=64,
          pre_conv_channels=[64, 32, 16, 8, 4],
          up_residuals=0,
          post_residuals=2):
        super(Head, self).__init__()
        pre_convs = []
        c0 = pre_conv_channels[0]
        pre_convs.append(ConvBlock(1, c0, kernel_size=3, padding=1))
        for _ in range(pre_residuals):
            pre_convs.append(ResnetBlock(c0, c0))

        for i in range(len(pre_conv_channels) -1):
            in_c = pre_conv_channels[i]
            out_c = pre_conv_channels[i + 1]
            pre_convs.append(ResnetBlock(in_c, out_c))
            for _ in range(pre_residuals):
                pre_convs.append(ResnetBlock(out_c, out_c))
        self.pre_conv = nn.Sequential(*pre_convs)

        up_layers = []
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            layer = UpsamplingLayer(in_channels, out_channels, residuals=up_residuals)
            up_layers.append(layer)
        self.upsampling = nn.Sequential(*up_layers)

        post_convs = []
        last_channels = channels[-1]
        for i in range(post_residuals):
            post_convs.append(ResnetBlock(last_channels, last_channels, one_d=True, kernel_size=5))
        self.post_conv = nn.Sequential(*post_convs)

    def forward(self, x):
        """
        forward pass
        Args:
            x (Tensor): B x C x T

        Returns:
            Tensor: B x C x (2^#channels * T)
        """
        x = x.unsqueeze(1) # reshape to [B x 1 x C x T]
        x = self.pre_conv(x)
        s1, _, _, s4 = x.shape
        x = x.reshape(s1, -1, s4)
        x = self.upsampling(x)
        x2 = self.post_conv(x)
        return x, x2


DEFAULT_LAYERS_PARAMS = [80, 128, 128, 64, 64, 32, 16, 8, 1]
class CNNVocoder(nn.Module):
    """CNN Vocoder

    Args:
        n_heads (int): Number of heads
        layer_channels (list): list of #channels of each layer
    """
    def __init__(self, n_heads=3, 
         layer_channels=DEFAULT_LAYERS_PARAMS,
         pre_conv_channels=[64, 32, 16, 8, 4],
         pre_residuals=64, 
         up_residuals=0,
         post_residuals=3):
        super(CNNVocoder, self).__init__()
        self.head = Head(layer_channels, 
                pre_conv_channels=pre_conv_channels, 
                pre_residuals=pre_residuals, up_residuals=up_residuals,
                post_residuals=post_residuals)
        self.linear = nn.Linear(layer_channels[-1], 1)
        self.act_fn = nn.Softsign()

    def forward(self, x):
        b = x.shape[0]
        pre, post = self.head(x)
       
        rs0 = self.linear(pre.transpose(1, 2))
        rs0 = self.act_fn(rs0).squeeze(-1)

        rs1 = self.linear(post.transpose(1, 2))
        rs1 = self.act_fn(rs1).squeeze(-1)
        return rs0, rs1 
