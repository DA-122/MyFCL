from vqvae import VQVAE, generate_samples
import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

print("test")

device = 0
state_dict = torch.load('./checkpoint/vqvae_best_1.pt')
model =  VQVAE(in_channels=3, embedding_dim = 64, num_embeddings = 128, img_size = 32)
model.load_state_dict(state_dict)
model.to(device)
# model = torch.load('vqvae_best.pt')

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = datasets.CIFAR100('../datasets', train=True, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle=True)
fixed_images, _ = next(iter(test_loader))
reconstruction = generate_samples(fixed_images, model, device)
show_img = torch.concat((fixed_images, reconstruction.cpu()), dim=2)
print(show_img.shape)
grid = make_grid(show_img, nrow=8, normalize=True)
plt.imshow(np.transpose(grid.numpy(),(1,2,0)),interpolation='nearest')
plt.show()