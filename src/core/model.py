"""
YOLOv4
============================================

From-scratch implementation of the YOLOv4 architecture proposed
in the paper "YOLOv4: Optimal Speed and Accuracy of Object Detection".

Main Components:
    - CNNBlock: Basic block (Conv + Batch Norm + Mish/LeakyReLU).
    - ResidualBlock: Residual connections used within CSPBlock.
    - CSPBlock: Cross Stage Partial block. Divides channels in half,
        passes one through residuals and the other through a bypass, then fuses them.
        Reduces backbone computation while maintaining learning capacity.
    - SPP: Spatial Pyramid Pooling. Applies MaxPooling at 3 different scales
        and concatenates the results to capture multi-scale context
        without reducing spatial dimensions.
    - PANet: Path Aggregation Network. Bidirectional neck that first passes
        semantic information (top-down) and then detail information
        (bottom-up), enriching features at each scale in both directions.
    - ScalePrediction: Detection head for each of the 3 scales.
    - YOLOv4: Main class that assembles the complete model.

    Input (Batch, 3, 416, 416)
        -> CSPDarknet53 (backbone)
        -> SPP
        -> PANet (bidirectional neck)
        -> 3 Outputs:
            (Batch, 3, 13, 13, num_classes+5)   large objects
            (Batch, 3, 26, 26, num_classes+5)   medium objects
            (Batch, 3, 52, 52, num_classes+5)   small objects

https://arxiv.org/abs/2004.10934
"""

import torch
import torch.nn as nn
from core.config import CONFIG

##### CONVOLUTION BLOCK #####
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, activation="leaky", **kwargs):
        # kwargs is a dictionary of parameters that can be passed to the function
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        # if bn is used, bias is removed because it already centers the data
        self.bn = nn.BatchNorm2d(out_channels)
        self.use_bn_act = bn_act

        if activation == "mish":
            self.activation = nn.Mish()
        else:
            self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        if self.use_bn_act:
            return self.activation(self.bn(self.conv(x)))
        else:
            return self.conv(x)

##### RESIDUAL BLOCK #####
class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1, activation="mish"):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1, padding=0, activation=activation), # 1x1 reduction
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1, activation=activation) # 3x3 expansion
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            # check if there is a residual connection
            x = x + layer(x) if self.use_residual else layer(x)
        return x


class CSPBlock(nn.Module):
    def __init__(self, in_channels, num_repeats):
        super().__init__()
        self.num_repeats = num_repeats
        # Channel division in half
        half_channels = in_channels // 2

        # Path 1: Simple convolution (the CSP "bypass")
        self.route_conv = CNNBlock(in_channels, half_channels, kernel_size=1, activation="mish")

        # Path 2: Passes through residual blocks
        self.main_conv = CNNBlock(in_channels, half_channels, kernel_size=1, activation="mish")
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(half_channels, activation="mish") for _ in range(num_repeats)]
        )
        self.post_res_conv = CNNBlock(half_channels, half_channels, kernel_size=1, activation="mish")

        # Finally concatenated: half + half = in_channels
        self.final_conv = CNNBlock(in_channels, in_channels, kernel_size=1, activation="mish")

    def forward(self, x):
        part1 = self.route_conv(x)
        part2 = self.post_res_conv(self.res_blocks(self.main_conv(x)))
        return self.final_conv(torch.cat([part1, part2], dim=1))

### Spatial Pyramid Pooling ###
class SPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = CNNBlock(in_channels, out_channels, kernel_size=1)
        self.m1 = nn.MaxPool2d(kernel_size=5,  stride=1, padding=2)
        self.m2 = nn.MaxPool2d(kernel_size=9,  stride=1, padding=4)
        self.m3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        # out_channels * 4 because we concatenate the 4 tensors
        self.conv2 = CNNBlock(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        out = torch.cat([x, self.m1(x), self.m2(x), self.m3(x)], dim=1)
        return self.conv2(out)

### Path Aggregation Network ###
class PANet(nn.Module):
    def __init__(self):
        super().__init__()

        # --- TOP-DOWN (FPN)(from 13x13 to 52x52) ---

        # Processes SPP output (13x13 x 512) and prepares it for upsampling
        self.td_conv1 = nn.Sequential(
            CNNBlock(512, 256, kernel_size=1),
            CNNBlock(256, 512, kernel_size=3, padding=1),
            CNNBlock(512, 256, kernel_size=1),
        )
        self.td_upsample1 = nn.Upsample(scale_factor=2)  # 13x13 -> 26x26

        # Fuses with route_medium (26x26): 256 + 512 = 768 input channels
        self.td_conv2 = nn.Sequential(
            CNNBlock(768, 256, kernel_size=1),
            CNNBlock(256, 512, kernel_size=3, padding=1),
            CNNBlock(512, 256, kernel_size=1),
        )
        self.td_upsample2 = nn.Upsample(scale_factor=2)  # 26x26 -> 52x52

        # Fuses with route_small (52x52): 256 + 256 = 512 input channels
        self.td_conv3 = nn.Sequential(
            CNNBlock(512, 128, kernel_size=1),
            CNNBlock(128, 256, kernel_size=3, padding=1),
            CNNBlock(256, 128, kernel_size=1),
        )

        # --- BOTTOM-UP (from 52x52 to 13x13) ---

        # Down from 52x52 to 26x26: 128 + 256 = 384 input channels
        self.bu_downsample1 = CNNBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.bu_conv1 = nn.Sequential(
            CNNBlock(512, 256, kernel_size=1),
            CNNBlock(256, 512, kernel_size=3, padding=1),
            CNNBlock(512, 256, kernel_size=1),
        )

        # Down from 26x26 to 13x13: 256 + 256 = 512 input channels
        self.bu_downsample2 = CNNBlock(256, 512, kernel_size=3, stride=2, padding=1)
        self.bu_conv2 = nn.Sequential(
            CNNBlock(768, 256, kernel_size=1),
            CNNBlock(256, 512, kernel_size=3, padding=1),
            CNNBlock(512, 256, kernel_size=1),
        )

    def forward(self, x_large, route_medium, route_small):
        # x_large is SPP output: 13x13 x 512

        # --- TOP-DOWN ---
        td1 = self.td_conv1(x_large)                                    # 13x13 x 256
        td1_up = self.td_upsample1(td1)                                 # 26x26 x 256
        td2 = self.td_conv2(torch.cat([td1_up, route_medium], dim=1))  # 26x26 x 256
        td2_up = self.td_upsample2(td2)                                 # 52x52 x 256
        td3 = self.td_conv3(torch.cat([td2_up, route_small], dim=1))   # 52x52 x 128

        # --- BOTTOM-UP ---
        bu1 = self.bu_downsample1(td3)                                  # 26x26 x 256
        bu1 = self.bu_conv1(torch.cat([bu1, td2], dim=1))              # 26x26 x 256
        bu2 = self.bu_downsample2(bu1)                                  # 13x13 x 512
        bu2 = self.bu_conv2(torch.cat([bu2, td1], dim=1))              # 13x13 x 256

        # Return the three scales for detection heads
        # td3: 52x52 x 128  -> small objects
        # bu1: 26x26 x 256  -> medium objects
        # bu2: 13x13 x 256  -> large objects
        return bu2, bu1, td3

### DETECTION HEAD ###
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
    
        self.pred = nn.Sequential(
            # final preprocessing: doubles size (according to YOLOv3 paper)
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),

            # and output with bn_act=False
            # out_channels = 3 * (num_classes + 5)
            #   --> 3: Because we use 3 anchors per cell
            #   --> 5: (x, y, w, h, object_probability)
            CNNBlock(2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # For the final fixed inference layer
            if m.bias is not None:
                # Initialize weights very small to not mutate strong features
                nn.init.normal_(m.weight, mean=0, std=0.01)

                # Initialize all bias to 0
                nn.init.constant_(m.bias, 0)

                # ONLY objectness (channel 0 of each of the 3 anchors) starts at -4.6
                num_channels = m.bias.shape[0]
                channels_per_anchor = num_channels // 3
                for a in range(3):
                    m.bias.data[a * channels_per_anchor] = -4.6
            else:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')


class YOLOv4(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels 
        self.backbone = self._create_conv_layers()
        self.spp = SPP(in_channels=1024, out_channels=512)
        self.neck = PANet()         
        self.head_large  = ScalePrediction(256, num_classes)   # 13x13
        self.head_medium = ScalePrediction(256, num_classes)   # 26x26
        self.head_small  = ScalePrediction(128, num_classes)   # 52x52  
        initialize_weights(self)

    def forward(self, x):
        route_small = None
        route_medium = None

        for layer in self.backbone:
            x = layer(x)
            if isinstance(layer, CSPBlock) and layer.num_repeats == 8:
                if route_small is None:
                    route_small = x   # first ["C", 8] -> 52x52 x 256
                else:
                    route_medium = x  # second ["C", 8] -> 26x26 x 512

        x = self.spp(x)  # 13x13 x 512

        out_large, out_medium, out_small = self.neck(x, route_medium, route_small)

        return [
            self.head_large(out_large),    # 13x13
            self.head_medium(out_medium),  # 26x26
            self.head_small(out_small),    # 52x52
        ]


    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in CONFIG:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels, out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                        activation="mish"
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):  # ["C", N]
                num_repeats = module[1]
                layers.append(CSPBlock(in_channels, num_repeats=num_repeats))

        return layers

# --- Sanity check ---
if __name__ == "__main__":
    num_classes = 20 # Example
    IMAGE_SIZE = 416 # Standard YOLO size
    # Simulate an image (Batch=2, Channels=3, Height=416, Width=416)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))

    model = YOLOv4(num_classes=num_classes)
    out = model(x)

    print("MODEL LOADED CORRECTLY!")
    print(f"Input: {x.shape}")
    print("Detected outputs (should be 3):")
    for i, o in enumerate(out):
        print(f" - Output {i+1}: {o.shape}")
        # Output should be: [2, 3, grid, grid, 25] (if classes=20, 20+5=25)