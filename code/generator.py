import torch
import torch.nn as nn

"""
Generates images of 64x64
"""

class Generator(nn.Module):
    def __init__(self, noise_size=100, spatial_size=4, depth=1024):
        super().__init__()

        self.projection = nn.Linear(noise_size, spatial_size * spatial_size * depth)
        self.batch_norm = nn.BatchNorm2d(depth)

        self.upsample = nn.Sequential(
            # block 1
            nn.ConvTranspose2d(1024, 512, 4, 2, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # block 2
            nn.ConvTranspose2d(512, 256, 4, 2, bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # block 3
            nn.ConvTranspose2d(256, 128, 4, 2, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # block 4
            nn.ConvTranspose2d(128, 3, 4, 2, bias=False, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, z): # z = [batch_size, noise_size]
        out = self.projection(z) # [batch_size, depth * spatial_size * spatial_size]
        out = out.view(-1, 1024, 4, 4) # [batch_size, depth, spatial_size, spatial_size]
        out = self.batch_norm(out)  
        out = torch.relu(out)

        return self.upsample(out)

if __name__ == "__main__":
    gen = Generator()

    batch_size = 128
    z = torch.rand(batch_size, 100) 
    initial_feature_map = gen(z)
    print(initial_feature_map.shape) # should be [128, 3, 64, 64]