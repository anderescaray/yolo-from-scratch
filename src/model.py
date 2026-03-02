"""
YOLOv4
============================================

Implementación desde cero de la arquitectura YOLOv4 propuesta
en el paper "YOLOv4: Optimal Speed and Accuracy of Object Detection".

Componentes Principales:
    - CNNBlock: Bloque básico (Conv + Batch Norm + Mish/LeakyReLU).
    - ResidualBlock: Conexiones residuales usadas dentro del CSPBlock.
    - CSPBlock: Bloque Cross Stage Partial. Divide los canales en dos mitades,
        pasa una por los residuales y la otra por un bypass, luego las fusiona.
        Reduce el cómputo del backbone manteniendo la capacidad de aprendizaje.
    - SPP: Spatial Pyramid Pooling. Aplica MaxPooling a 3 escalas distintas
        y concatena los resultados para capturar contexto multi-escala
        sin reducir el tamaño espacial.
    - PANet: Path Aggregation Network. Neck bidireccional que primero baja
        información semántica (top-down) y luego sube información de detalle
        (bottom-up), enriqueciendo las features de cada escala en ambas direcciones.
    - ScalePrediction: Cabezal de detección para cada una de las 3 escalas.
    - YOLOv4: Clase principal que ensambla el modelo completo.

    Input (Batch, 3, 416, 416)
        -> CSPDarknet53 (backbone)
        -> SPP
        -> PANet (neck bidireccional)
        -> 3 Salidas:
            (Batch, 3, 13, 13, num_classes+5)   objetos grandes
            (Batch, 3, 26, 26, num_classes+5)   objetos medianos
            (Batch, 3, 52, 52, num_classes+5)   objetos pequeños

https://arxiv.org/abs/2004.10934
"""

import torch
import torch.nn as nn
from config import CONFIG

##### BLOQUE CONVOLUCIÓN #####
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, activation="leaky", **kwargs):
        #kwargs es un diccionario de parámetros que se pueden pasar a la función
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        # si se usa bn se quita el bias porque ya centra los datos
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

##### BLOQUE RESIDUAL #####
class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1, activation="mish"):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1, padding=0, activation=activation), # reducción 1x1
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1, activation=activation) # expansión 3x3
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            # se verifica si hay conexión residual
            x = x + layer(x) if self.use_residual else layer(x)
        return x


class CSPBlock(nn.Module):
    def __init__(self, in_channels, num_repeats):
        super().__init__()
        self.num_repeats = num_repeats
        # División de canales a la mitad
        half_channels = in_channels // 2
        
        # Camino 1: Convolución simple (el "bypass" de CSP)
        self.route_conv = CNNBlock(in_channels, half_channels, kernel_size=1, activation="mish")
        
        # Camino 2: Pasa por los bloques residuales
        self.main_conv = CNNBlock(in_channels, half_channels, kernel_size=1, activation="mish")
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(half_channels, activation="mish") for _ in range(num_repeats)]
        )
        self.post_res_conv = CNNBlock(half_channels, half_channels, kernel_size=1, activation="mish")
        
        # Al final se concatenan: half + half = in_channels
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
        # out_channels * 4 porque concatenamos 4 tensores
        self.conv2 = CNNBlock(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        out = torch.cat([x, self.m1(x), self.m2(x), self.m3(x)], dim=1)
        return self.conv2(out)

### DETECTION HEAD ###
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
    
        self.pred = nn.Sequential(
            # preprocesamiento final: duplica el tamaño (según paper YOLOv3)
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            
            # y la salida con bn_act=False
            # out_channels = 3 * (num_classes + 5)
            #   --> 3: Porque usamos 3 anclas por celda
            #   --> 5: (x, y, w, h, probabilidad_objeto)
            CNNBlock(2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )

### Path Aggregation Network ###
class PANet(nn.Module):
    def __init__(self):
        super().__init__()

        # --- TOP-DOWN (de 13x13 hacia 52x52) ---

        # Procesa la salida del SPP (13x13 x 512) y la prepara para subir
        self.td_conv1 = nn.Sequential(
            CNNBlock(512, 256, kernel_size=1),
            CNNBlock(256, 512, kernel_size=3, padding=1),
            CNNBlock(512, 256, kernel_size=1),
        )
        self.td_upsample1 = nn.Upsample(scale_factor=2)  # 13x13 -> 26x26

        # Fusiona con route_medium (26x26): 256 + 512 = 768 canales entrada
        self.td_conv2 = nn.Sequential(
            CNNBlock(768, 256, kernel_size=1),
            CNNBlock(256, 512, kernel_size=3, padding=1),
            CNNBlock(512, 256, kernel_size=1),
        )
        self.td_upsample2 = nn.Upsample(scale_factor=2)  # 26x26 -> 52x52

        # Fusiona con route_small (52x52): 256 + 256 = 512 canales entrada
        self.td_conv3 = nn.Sequential(
            CNNBlock(512, 128, kernel_size=1),
            CNNBlock(128, 256, kernel_size=3, padding=1),
            CNNBlock(256, 128, kernel_size=1),
        )

        # --- BOTTOM-UP (de 52x52 hacia 13x13) ---

        # Baja de 52x52 a 26x26: 128 + 256 = 384 canales entrada
        self.bu_downsample1 = CNNBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.bu_conv1 = nn.Sequential(
            CNNBlock(512, 256, kernel_size=1),
            CNNBlock(256, 512, kernel_size=3, padding=1),
            CNNBlock(512, 256, kernel_size=1),
        )

        # Baja de 26x26 a 13x13: 256 + 256 = 512 canales entrada
        self.bu_downsample2 = CNNBlock(256, 512, kernel_size=3, stride=2, padding=1)
        self.bu_conv2 = nn.Sequential(
            CNNBlock(768, 256, kernel_size=1),
            CNNBlock(256, 512, kernel_size=3, padding=1),
            CNNBlock(512, 256, kernel_size=1),
        )

    def forward(self, x_large, route_medium, route_small):
        # x_large es la salida del SPP: 13x13 x 512

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

        # Devolvemos las tres escalas para las cabezas de detección
        # td3: 52x52 x 128  -> objetos pequeños
        # bu1: 26x26 x 256  -> objetos medianos
        # bu2: 13x13 x 256  -> objetos grandes
        return bu2, bu1, td3

class YOLOv4(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels 
        self.backbone = self._create_conv_layers()
        self.spp = SPP(in_channels=1024, out_channels=512)
        self.neck = PANet()          # lo definimos a continuación
        self.head_large  = ScalePrediction(256, num_classes)   # 13x13
        self.head_medium = ScalePrediction(256, num_classes)   # 26x26
        self.head_small  = ScalePrediction(128, num_classes)   # 52x52  

    def forward(self, x):
        route_small = None
        route_medium = None

        for layer in self.backbone:
            x = layer(x)
            if isinstance(layer, CSPBlock) and layer.num_repeats == 8:
                if route_small is None:
                    route_small = x   # primer ["C", 8] -> 52x52 x 256
                else:
                    route_medium = x  # segundo ["C", 8] -> 26x26 x 512

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
    num_classes = 20 # Ejemplo
    IMAGE_SIZE = 416 # Tamaño estándar YOLO
    # Simulamos una imagen (Batch=2, Canales=3, Alto=416, Ancho=416)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    
    model = YOLOv4(num_classes=num_classes)
    out = model(x)
    
    print("¡MODELO CARGADO CORRECTAMENTE!")
    print(f"Entrada: {x.shape}")
    print("Salidas detectadas (deben ser 3):")
    for i, o in enumerate(out):
        print(f" - Salida {i+1}: {o.shape}")
        # La salida debe ser: [2, 3, grid, grid, 25] (si clases=20, 20+5=25)