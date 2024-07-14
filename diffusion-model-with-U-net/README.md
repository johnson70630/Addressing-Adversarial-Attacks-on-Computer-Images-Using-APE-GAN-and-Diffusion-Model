# diffusion-model-with-U-net
Build a simple U-net Diffusion Model using PyTorch

## [Diffusion model](https://github.com/johnson70630/diffusion-model-with-U-net/blob/main/diffusion_U_net.ipynb)

### Forward process
Adding noise to the original images and sampling images with different levels of added noise

```ruby
def corrupt(x, amount):
  noise = torch.rand_like(x)
  amount = amount.reshape(-1, 1, 1, 1)
  return x*(1-amount) + noise*amount, noise*amount
```
```ruby
# Plotting the input data
fig, axs = plt.subplots(2, 1, figsize=(12, 5))
axs[0].set_title('Input data')
axs[0].imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')

# Adding noise
# x.shape => 8, 1, 32, 32
amount = torch.linspace(0, 1, x.shape[0]) # Left to right -> more corruption
noised_x, noise = corrupt(x, amount)

# Plotting the noised version
axs[1].set_title('Corrupted data (-- amount increases -->)')
axs[1].imshow(torchvision.utils.make_grid(noised_x)[0], cmap='Greys');
```

### Reverse process
Using U-net to restore images, bringing the images back to their original noise-free appearance
```ruby
class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.double_conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels , out_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
  def forward(self, x):
    return self.double_conv(x)
```
```ruby
class Unet(nn.Module):
  def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
    super().__init__()
    self.ups = nn.ModuleList()
    self.downs = nn.ModuleList()

    for feature in features:
      self.downs.append(DoubleConv(in_channels, feature))
      in_channels = feature

    for feature in reversed(features):
      self.ups.append(nn.ConvTranspose2d(feature*2, feature, 2, 2))
      self.ups.append(DoubleConv(feature*2, feature))

    self.bottleneck = DoubleConv(features[-1], features[-1]*2)
    self.final_conv = nn.Conv2d(features[0], out_channels, 1)

  def forward(self, x):
    skip_connections = []
    for down in self.downs:
      x = down(x) # 3 -> 64 -> 128 -> 256 -> 512
      skip_connections.append(x)
      x = F.max_pool2d(x, (2, 2))
    x = self.bottleneck(x)
    skip_connections.reverse()

    for i in range(0, len(self.ups), 2):
      x = self.ups[i](x)
      skip_connection = skip_connections[i//2]
      concat_x = torch.cat((skip_connection, x), dim=1)
      x = self.ups[i+1](concat_x)

    return self.final_conv(x)
```

### U-net
Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox:
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

![image](https://github.com/johnson70630/diffusion-model-with-U-net/assets/104968059/b8b02dd0-8cb3-4737-8f76-717658d2e13e)
