"""Tres modelos base de segmentación semántica binaria (tornillo / fondo).

Los tres comparten el MISMO codificador (4 bloques conv-BN-ReLU x2 + max-pooling,
anchos c, 2c, 4c, 8c y cuello de botella 16c). Solo cambia cómo recuperan la
resolución, que es el sesgo inductivo que los diferencia:

- FCN (estilo FCN-8s, "FCN sencilla"): puntajes 1x1 en tres escalas, suma y
  sobremuestreo bilineal fijo. Sin decodificador aprendido.
- SegNet ("codificador-decodificador convolucional básico"): el decodificador
  desagrupa con los índices del max-pooling y refina con convoluciones; no copia
  mapas del codificador.
- U-Net ("U-Net reducida"): el decodificador concatena los mapas completos del
  codificador (conexiones de salto).

Salida: un logit por píxel (1 canal), mismo alto y ancho que la entrada.
La entrada debe tener alto y ancho divisibles por 16 (720x1280 y 512x512 lo son).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bloque(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class Codificador(nn.Module):
    def __init__(self, c: int = 16):
        super().__init__()
        self.anchos = [c, 2 * c, 4 * c, 8 * c]
        cin = 3
        self.etapas = nn.ModuleList()
        for a in self.anchos:
            self.etapas.append(bloque(cin, a))
            cin = a
        self.cuello = bloque(8 * c, 16 * c)

    def forward(self, x, con_indices: bool = False):
        saltos, indices = [], []
        for etapa in self.etapas:
            x = etapa(x)
            saltos.append(x)
            if con_indices:
                x, idx = F.max_pool2d(x, 2, return_indices=True)
                indices.append(idx)
            else:
                x = F.max_pool2d(x, 2)
        return self.cuello(x), saltos, indices


class UNet(nn.Module):
    def __init__(self, c: int = 16):
        super().__init__()
        self.enc = Codificador(c)
        anchos = self.enc.anchos[::-1]  # 8c, 4c, 2c, c
        self.subir = nn.ModuleList()
        self.dec = nn.ModuleList()
        cin = 16 * c
        for a in anchos:
            self.subir.append(nn.ConvTranspose2d(cin, a, 2, stride=2))
            self.dec.append(bloque(2 * a, a))
            cin = a
        self.salida = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        x, saltos, _ = self.enc(x)
        for subir, dec, s in zip(self.subir, self.dec, saltos[::-1]):
            x = dec(torch.cat([subir(x), s], dim=1))
        return self.salida(x)


class SegNet(nn.Module):
    def __init__(self, c: int = 16):
        super().__init__()
        self.enc = Codificador(c)
        anchos = self.enc.anchos[::-1]  # 8c, 4c, 2c, c
        self.reducir = nn.Conv2d(16 * c, 8 * c, 1)  # cuello 16c -> 8c para desagrupar con índices de la etapa 4
        self.dec = nn.ModuleList()
        for i, a in enumerate(anchos):
            self.dec.append(bloque(a, anchos[i + 1] if i + 1 < len(anchos) else c))
        self.salida = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        x, saltos, indices = self.enc(x, con_indices=True)
        x = self.reducir(x)
        for dec, idx, s in zip(self.dec, indices[::-1], saltos[::-1]):
            x = F.max_unpool2d(x, idx, 2, output_size=s.shape[-2:])
            x = dec(x)
        return self.salida(x)


class FCN(nn.Module):
    """FCN-8s sencilla: puntajes en 1/16, 1/8 y 1/4 (por la profundidad de este codificador)."""

    def __init__(self, c: int = 16):
        super().__init__()
        self.enc = Codificador(c)
        self.p16 = nn.Conv2d(16 * c, 1, 1)
        self.p8 = nn.Conv2d(8 * c, 1, 1)
        self.p4 = nn.Conv2d(4 * c, 1, 1)

    def forward(self, x):
        alto, ancho = x.shape[-2:]
        b, saltos, _ = self.enc(x)
        # saltos[i] es la salida de la etapa i antes del pooling: [2] está a 1/4 y [3] a 1/8.
        y = self.p16(b)
        y = F.interpolate(y, size=saltos[3].shape[-2:], mode="bilinear", align_corners=False) + self.p8(saltos[3])
        y = F.interpolate(y, size=saltos[2].shape[-2:], mode="bilinear", align_corners=False) + self.p4(saltos[2])
        return F.interpolate(y, size=(alto, ancho), mode="bilinear", align_corners=False)


MODELOS = {"fcn": FCN, "segnet": SegNet, "unet": UNet}


def crear(nombre: str, c: int = 16) -> nn.Module:
    return MODELOS[nombre](c)


def n_parametros(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def profundidad(m: nn.Module) -> int:
    """Capas con pesos (convoluciones y transpuestas) en el camino más largo."""
    return sum(1 for x in m.modules() if isinstance(x, (nn.Conv2d, nn.ConvTranspose2d)))
