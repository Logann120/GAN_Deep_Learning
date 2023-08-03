import concurrent.futures
import argparse
import os
import random
import numpy as np
from torchsummary import summary

import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import torch.utils.data.dataloader


import torch.nn as nn
import torch
import torch.cuda

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=66, help="size of the batches")
parser.add_argument("--lrg", type=float, default=0.00012, help="adam: learning rate")
parser.add_argument("--lrd", type=float, default=0.0003, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=(96, 128), help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval betwen image samples")
parser.add_argument("--dir", type=str, default="images", help="directory to save images")
parser.add_argument("--v", action='store_true', help="visualize model")
parser.add_argument("--model", type=str, default="model.th", help="model to load")
opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.dir, exist_ok=False)
except FileExistsError:
    # Remove directory and all files
    for filename in os.listdir(opt.dir):
        os.remove(os.path.join(opt.dir, filename))
    os.rmdir(opt.dir)
    # Create directory
    os.makedirs(opt.dir, exist_ok=False)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_shape = (opt.channels, opt.img_size[0], opt.img_size[1])


class CustomDataset():
    def __init__(self, root_dir, transform: transforms.Compose):
        self.root_dir = root_dir
        self.image_files = os.listdir(root_dir)
        self.transform = transform
        self.images = self._load_images()

    def _load_image(self, img_name):
        image = Image.open(img_name).convert("L").copy()
        image = TF.center_crop(TF.resize(image,  # type: ignore
                                         [3300,
                                          4400],
                                         interpolation=transforms.InterpolationMode.NEAREST),
                               output_size=[int(3300 * 0.7),
                                            int(4400 * 0.7)])
        return self.transform(image)

    def _load_images(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._load_image, os.path.join(self.root_dir, img_name))
                       for img_name in self.image_files]
            images = [f.result() for f in futures]
        self.transform = get_transforms()
        return images

    def __len__(self):
        return len(self.image_files)

    def shuffle(self):
        transformed_images = torch.stack(self.images)
        return transformed_images[torch.randperm(transformed_images.size(0))]


def save_model(model, save_path):
    from torch import save
    return save(model.state_dict(), save_path)


def load_model(model_path='model.th'):
    from torch import load
    r = Generator()
    r.load_state_dict(load(model_path, map_location='cpu'))
    return r


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = []
            layers.append(nn.Linear(in_feat, out_feat, bias=normalize))
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.25, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.25, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img = img.view(img.size(0), *img_shape) + torch.as_tensor(np.random.normal(0.,
                                                                                   0.1, (img.size(0), *img_shape)), dtype=torch.float32, device=device)
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


def visualize_model(fname='model.th'):
    generator = load_model('models/' + fname)
    generator = generator.to(device)
    generator.eval()
    eval_inputs = torch.as_tensor(np.random.normal(0., 1., (30, opt.latent_dim)), dtype=torch.float32, device=device)
    eval_imgs = generator(eval_inputs)
    save_image([img.transpose(2, 1) for img in eval_imgs], 'test_img.png', nrow=5, normalize=False)
    exit(0)


if opt.v:
    visualize_model(opt.model)

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()


print(summary(generator, (66, 100), batch_dim=None))
print(summary(discriminator, (66, 1, 96 * 128), batch_dim=None))
if torch.cuda.is_available():
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    adversarial_loss = adversarial_loss.to(device)


def get_transforms():
    data_transform = transforms.Compose([
        transforms.Lambda(lambda x: TF.adjust_contrast(x, random.uniform(1.1, 1.5))),
        transforms.Lambda(lambda x: TF.adjust_brightness(x, random.uniform(1.1, 1.5))),
        transforms.Lambda(lambda x: TF.adjust_sharpness(x, random.uniform(1.1, 3.0))),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((96, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return data_transform


def plot_losses_and_save_plot(d_losses, g_losses, save_path='losses.png'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(d_losses[::3], label="D")
    plt.plot(g_losses[::3], label="G")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


dataset = CustomDataset(root_dir="Hazel_Train", transform=get_transforms())
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lrg, betas=(opt.b1, opt.b2))

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lrd, betas=(opt.b1, opt.b2))

# ----------
#  Training
# ----------

d_losses, g_losses = [], []
eval_inputs = torch.as_tensor(np.random.normal(0., 1., (25, opt.latent_dim)), dtype=torch.float32, device=device)
i = 0
for epoch in range(1, opt.n_epochs + 1):
    imgs = dataset.shuffle()
    # Adversarial ground truths
    valid = torch.full((imgs.size(0), 1), 1.0, dtype=torch.float32).to(device)
    fake = torch.full((imgs.size(0), 1), 0.0, dtype=torch.float32).to(device)

    # Configure input
    real_imgs = imgs.to(device)  # type: ignore

    # -----------------
    #  Train Generator
    # -----------------
    optimizer_G.zero_grad()

    # Sample noise as generator input
    z = torch.as_tensor(np.random.normal(0., 1.1, (opt.batch_size, opt.latent_dim)), dtype=torch.float32, device=device)

    # Generate a batch of images
    gen_imgs = generator(z)

    # Loss measures generator's ability to fool the discriminator
    g_loss = adversarial_loss(discriminator(gen_imgs), valid)

    g_loss.backward()
    optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()

    # Measure discriminator's ability to classify real from generated samples
    real_loss = adversarial_loss(discriminator(real_imgs), valid)
    fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
    d_loss = (real_loss + fake_loss) / 2

    d_loss.backward()
    optimizer_D.step()

    d_losses.append(d_loss.detach().item())
    g_losses.append(g_loss.detach().item())

    if (epoch + 1) % opt.sample_interval == 0:
        print(
            "[Epoch %d/%d] [D loss: %.2f] [G loss: %.2f]" % ((epoch + 1), opt.n_epochs, d_loss.detach().item(), g_loss.detach().item()))

        generator.eval()
        with torch.no_grad():
            eval_imgs = generator(eval_inputs)
        generator.train()
        save_image(eval_imgs, opt.dir +
                   "/%d.png" % (i), nrow=5, normalize=False)
        if not os.path.exists('models'):
            os.mkdir('models')
        save_model(generator, os.path.join('models', 'generator' + str(i) + '.th'))
    i += 1

plot_losses_and_save_plot(d_losses, g_losses)


def make_gif_from_imgs(imgs_path):
    import imageio
    import os
    images = []
    for filename in sorted(os.listdir(imgs_path)):
        images.append(imageio.imread(os.path.join(imgs_path, filename)))
    imageio.mimsave('movie.gif', images, duration=200)


def show_gif(gif_path):
    from IPython.core.display import Image
    return Image(filename=gif_path)


make_gif_from_imgs(opt.dir)
show_gif('movie.gif')
