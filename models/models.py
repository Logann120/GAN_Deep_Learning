import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(96, 128)):
        super(Generator, self).__init__()
        self.img_shape = torch.as_tensor(img_shape)

        def block(in_feat, out_feat, normalize=True):
            layers = []
            layers.append(nn.Linear(in_feat, out_feat, bias=normalize))
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, eps=1e-1))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 256, normalize=False),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(torch.prod(self.img_shape))),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return F.tanh(img)


class Discriminator(nn.Module):
    def __init__(self, img_shape=(96, 128)):
        super(Discriminator, self).__init__()
        self.img_shape = torch.as_tensor(img_shape)
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.55, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.55, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        # img = img.view(-1, 96, 128) + gaussian_blur(img.view(-1, 96, 128), 3, self.gaussian_stddev)  # type: ignore
        validity = self.model(img.view(img.size(0), -1))
        return validity
