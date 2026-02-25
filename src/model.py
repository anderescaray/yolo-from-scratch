"""
YOLOv3 
============================================

Implementación desde cero de la arquitectura YOLOv3 propuesta
en el paper "YOLOv3: An Incremental Improvement".

Componentes Principales:
    - CNNBlock: Bloque básico (Conv + Batch Norm + Leaky ReLU).
    - ResidualBlock: Conexiones residuales para el backbone Darknet-53.
    - ScalePrediction: Cabezal de detección para las 3 escalas.
    - YOLOv3: Clase principal que ensambla el modelo usando FPN (Feature Pyramid Network).


    Input (Batch, 3, 416, 416) -> Darknet-53 -> FPN (Upsampling + Concatenation) -> 
    3 Salidas [(Batch, 3, 13, 13, 85), (Batch, 3, 26, 26, 85), (Batch, 3, 52, 52, 85)]

https://arxiv.org/abs/1804.02767
"""
import torch
import torch.nn as nn
from config import CONFIG

##### BLOQUE CONVOLUCIÓN #####
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        #kwargs es un diccionario de parámetros que se pueden pasar a la función
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        # si se usa bn se quita el bias porque ya centra los datos
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)

##### BLOQUE RESIDUAL #####
class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1, padding=0), # reducción 1x1
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1) # expansión 3x3
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            # se verifica si hay conexión residual
            x = x + layer(x) if self.use_residual else layer(x)
        return x

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


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        # Creamos todas las capas leyendo la lista CONFIG
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []          # Aquí guardaremos las 3 salidas (predicciones)
        route_connections = [] # Memoria para guardar capas y concatenarlas luego

        for layer in self.layers:
            
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            # 
            x = layer(x)

            # (Guardar los mapas de características para concatenar luego)
            # En YOLOv3 original, se concatenan las capas después de los bloques residuales 8
            # Este es un truco para identificar esas capas específicas observando sus canales:
            # - Después del bloque residual 8 (primero) -> Canales son 256
            # - Después del bloque residual 8 (segundo) -> Canales son 512
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            
            elif isinstance(layer, nn.Upsample):
                # Recuperamos la última capa guardada en memoria (la del bloque residual anterior)
                # y la concatenamos con la actual para recuperar nitidez
                # dim=1 significa concatenar en la dimensión de los canales (profundidad)
                x = torch.cat([x, route_connections.pop()], dim=1)

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in CONFIG:
            # si es una tupla -> conv
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            # si es una lista -> Bloque Residual
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))

            # Salida (S) o Upsample (U)
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        # Antes de predecir con YOLO se suele hacer un bloque residual extra y una conv
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes)
                    ]
                    in_channels = in_channels // 2
                
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    # Al concatenar con una capa anterior, los canales aumentan.
                    # En YOLOv3 siempre se concatena con una capa que tiene el triple de tamaño,
                    # así que multiplicamos por 3
                    in_channels = in_channels * 3

        return layers

# --- PRUEBA DE FUEGO (SANITY CHECK) ---
if __name__ == "__main__":
    num_classes = 20 # Ejemplo
    IMAGE_SIZE = 416 # Tamaño estándar YOLO
    # Simulamos una imagen (Batch=2, Canales=3, Alto=416, Ancho=416)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    
    model = YOLOv3(num_classes=num_classes)
    out = model(x)
    
    print("¡MODELO CARGADO CORRECTAMENTE!")
    print(f"Entrada: {x.shape}")
    print("Salidas detectadas (deben ser 3):")
    for i, o in enumerate(out):
        print(f" - Salida {i+1}: {o.shape}")
        # La salida debe ser: [2, 3, grid, grid, 25] (si clases=20, 20+5=25)