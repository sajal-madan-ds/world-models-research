from torch import nn


class ImageEncoder(nn.Module):
    """A deliberately tiny two-frame encoder, not a ViT or official JEPA."""

    def __init__(self, latent=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, latent),
        )

    def forward(self, x):
        return self.net(x)


class ImageDecoder(nn.Module):
    def __init__(self, latent=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, 128),
            nn.GELU(),
            nn.Linear(128, 512),
            nn.Sigmoid(),
            nn.Unflatten(1, (2, 16, 16)),
        )

    def forward(self, x):
        return self.net(x)
