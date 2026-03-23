import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.downsample = nn.Sequential(
            # block 1
            nn.Conv2d(3, 64, 4, 2, bias=False, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # block 2
            nn.Conv2d(64, 128, 4, 2, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # block 3
            nn.Conv2d(128, 256, 4, 2, bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # block 4
            nn.Conv2d(256, 512, 4, 2, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.downsample(z)
        return out

if __name__ == "__main__":
    dis = Discriminator()
    x = torch.rand((128, 3, 64, 64))
    print(dis(x).shape) # should be [128, 1, 1, 1]