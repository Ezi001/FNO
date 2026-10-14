import torch
from models.unet_torch import UNet2D  # Adjust to your actual module path

model = UNet2D(
    input_channels=1,
    channel_mult=(64, 128, 256),
    use_condition=False,
)

# One SSH field: (batch, channels, height, width)
x = torch.randn(2, 1, 211, 211)

# Optional time values, one per batch sample
t = torch.rand(2)

with torch.no_grad():
    y = model(x, t)

print("Input shape:", x.shape)
print("Output shape:", y.shape)