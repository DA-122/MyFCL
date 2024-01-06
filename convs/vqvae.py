import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Latent 离散化module
class VectorQuantizer(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # 计算latent和embedding之间的距离
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # 获得距离最小的embedding idx
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # 转化成 one hot 编码
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # latent 量化
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # 计算VQ Loss
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]
    

class ResidualLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.resblock(input)


class VQVAE(nn.Module):
    def __init__(self,
                in_channels: int,
                embedding_dim: int,
                num_embeddings: int,
                hidden_dims: list = None,
                beta: float = 0.25,
                img_size: int = 64) -> None:
        super(VQVAE, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]

        # 升通道数
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(3):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim,
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        self.beta)
        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim,
                          hidden_dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.LeakyReLU())
        )

        for _ in range(3):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.LeakyReLU())
            )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   out_channels=3,
                                   kernel_size=4,
                                   stride=2, padding=1),
                nn.Tanh()))

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        result = self.encoder(input)
        return result
        # return [result]


    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder(z)
        return result

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # encoding = self.encode(input)[0]
        encoding = self.encode(input)
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return self.decode(quantized_inputs), input, vq_loss

    def loss_function(self, input, recons, vq_loss) -> torch.Tensor:
        recons_loss = F.mse_loss(recons, input)
        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}




batch_size = 64
num_epochs = 50
lr = 0.001
beta = 0.25
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_filename = 'vqvae'
# 等于encode output channel
embedding_dim = 64
# embedding 数量
num_embedding = 128




def train(data_loader, model, optimizer, device, beta, scheduler):
    model.train()
    model.to(device)
    for images, _ in data_loader:
        images = images.to(device)

        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x = model(images)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + beta * loss_commit
        loss.backward()

        optimizer.step()
    scheduler.step()


def test(data_loader, model, device):
    model.eval()
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        for images, _ in data_loader:
            images = images.to(device)
            x_tilde, z_e_x, z_q_x = model(images)
            loss_recons += F.mse_loss(x_tilde, images)
            loss_vq += F.mse_loss(z_q_x, z_e_x)

        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)
    return loss_recons.item(), loss_vq.item()


def generate_samples(images, model, device):
    with torch.no_grad():
        images = images.to(device)
        x_tilde, _, _ = model(images)
    return x_tilde

if __name__ == '__main__':
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR100('../datasets', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100('../datasets', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle=True)

    model =  VQVAE(in_channels=3, embedding_dim = 64, num_embeddings = 128, img_size = 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    fixed_images, _ = next(iter(test_loader))   
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1, last_epoch=-1)

    for epoch in range(num_epochs):
        train(train_loader, model, optimizer, device, beta, scheduler)
        loss, _ = test(test_loader, model, device)

        print('epoch {}, loss: {}'.format(epoch + 1, loss))

        reconstruction = generate_samples(fixed_images, model, device)
        grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            # with open('vqvae_{0}/best.pt'.format(save_filename), 'wb') as f:
                # torch.save(model.state_dict(), f)
            torch.save(model.state_dict(), 'vqvae_best_1_cifar100.pt')
        if epoch == num_epochs - 1:
            print("train complete , best loss: {}".format(best_loss))
        # with open('vqvae_{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
        #     torch.save(model.state_dict(), f) 
        # torch.save(model.state_dict, 'vqvae_model_{}.pt')



