import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur
from torchvision.transforms import Compose
from PIL import Image
import matplotlib.pyplot as plt
from torchsummary import summary

device = torch.device("cuda")

normal_samples = torch.distributions.Normal(torch.full((1, 100), 0.0), torch.full((1, 100), 1.0))

# Custom dataset class for loading and preprocessing images


class CustomDataset():
    def __init__(self, root_dir, transform: Compose):
        self.root_dir = root_dir
        self.image_files = os.listdir(root_dir)
        self.transform = transform
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for i in range(len(self.image_files)):
            img_name = os.path.join(self.root_dir, self.image_files[i])
            image = Image.open(img_name).convert("L").copy()
            # image = self.transform(image)
            images.append(TF.center_crop(TF.resize(image,  # type: ignore
                                                   [3300,
                                                    4400],
                                                   interpolation=transforms.InterpolationMode.NEAREST_EXACT),
                                         output_size=[int(3300 * 0.7),
                                                      int(4400 * 0.7)]))
        return images

    def __len__(self):
        return len(self.image_files)

    def shuffle(self):
        transformed_images = torch.stack([self.transform(img) for img in self.images])
        return transformed_images[torch.randperm(transformed_images.size()[0])]

# Generator model


class Generator(nn.Module):
    class UpBlock(nn.Module):
        def __init__(self, in_channels, out_channels, skip_connect=True):
            super().__init__()
            self.network = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, 1, padding='same', bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
            self.skip = skip_connect
            if skip_connect:
                self.skip_connect = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, 1, padding='same'),
                                                  nn.Upsample(scale_factor=2, mode='nearest'))

        def forward(self, x):
            out = self.network(x)
            if self.skip:
                out = out + self.skip_connect(x)
            return out

    def __init__(self, image_dim):
        super().__init__()
        self.image_dim = image_dim
        self.dense = nn.Linear(100, 256 * 6 * 8, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        # self.feat_map = nn.Conv2d(in_channels=1,out_channels=256,kernel_size=5,stride=1,padding='same')
        # Upsampling layers with skip connections
        self.upsample_1 = self.UpBlock(256, 64)
        self.conv1 = nn.Conv2d(64, 64, 3, 1, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.upsample_2 = self.UpBlock(64, 16)
        self.conv2 = nn.Conv2d(16, 16, 5, 1, padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.upsample_3 = self.UpBlock(16, 4)
        self.conv3 = nn.Conv2d(4, 4, 7, 1, padding='same', bias=False)
        self.bn4 = nn.BatchNorm2d(4)
        self.upsample_4 = self.UpBlock(4, 1)

        # Final convolution layer to generate the output image
        # self.classifier = nn.Conv2d(32, 1, 1, 1, padding='same')

    def forward(self, z: torch.Tensor):
        # x: torch.Tensor = self.feat_map(z.view(-1, 3, 6, 8))
        input = self.dense(z).view(-1, 256, 6, 8)
        input = self.bn1(input)
        # print('Input Shape:', input.shape)
        skip1 = input
        up1_out = self.upsample_1(F.relu(input))
        # print('Up1 Shape:', up1_out.shape)
        conv1_out = self.conv1(F.relu(up1_out))
        conv1_out = self.bn2(conv1_out)
        up2_out = self.upsample_2(F.relu(conv1_out)) + skip1.view(-1, 16, 24, 32)
        # print('Up2 Shape:', up2_out.shape)
        skip2 = up2_out
        conv2_out = self.conv2(F.relu(up2_out))
        conv2_out = self.bn3(conv2_out)
        up3_out = self.upsample_3(F.relu(conv2_out))
        # print('Up3 Shape:', up3_out.shape)
        conv3_out = self.conv3(F.relu(up3_out))
        conv3_out = self.bn4(conv3_out)
        up4_out = self.upsample_4(F.relu(conv3_out)) + skip2.view(-1, 1, 96, 128)

        # print('Up4 Shape:', up4_out.shape)
        return F.tanh(up4_out)


class D(nn.Module):
    class DownBlock(nn.Module):
        def __init__(self, in_channels, out_channels, skip_connect=True):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.skip = skip_connect
            if skip_connect:
                self.skip_connect = nn.Conv2d(in_channels, out_channels, 1, 2)

        def forward(self, x):
            out = self.network(x)
            if self.skip:
                out = out + self.skip_connect(x)
            return out

    def __init__(self, image_dim):
        super().__init__()
        self.image_dim = image_dim  # (96, 128)

        self.downsample_1 = self.DownBlock(1, 32, skip_connect=False)
        self.downsample_2 = self.DownBlock(32, 64, skip_connect=False)
        self.downsample_3 = self.DownBlock(64, 128, skip_connect=False)
        self.downsample_4 = self.DownBlock(128, 256, skip_connect=False)
        self.classifier = nn.Sequential(nn.Conv2d(256, 1, (6, 8), 1, padding='valid'), nn.Conv2d(1, 1, 1, 1))

    def forward(self, z: torch.Tensor):
        z = gaussian_blur(z, kernel_size=[3, 3])
        x = F.leaky_relu(self.downsample_1(z), negative_slope=0.1)
        x = F.dropout(x, 0.5)
        x = F.leaky_relu(self.downsample_2(x), negative_slope=0.1)
        x = F.dropout(x, 0.5)
        x = F.leaky_relu(self.downsample_3(x), negative_slope=0.1)
        x = F.dropout(x, 0.5)
        x = F.leaky_relu(self.downsample_4(x), negative_slope=0.1)
        x = F.dropout(x, 0.5)

        x = self.classifier(x)
        return x.view(-1, 1)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0., 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1., 0.02)
        nn.init.constant_(m.bias.data, 0.)
    else:
        pass


class GAN_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = Generator(image_dim=(96, 128))
        self.generator.apply(weights_init)
        self.discriminator = D(image_dim=(96, 128))
        self.discriminator.apply(weights_init)

    def forward(self, z):
        fake_images = self.generator(z)
        fake_outputs = self.discriminator(fake_images)
        return fake_outputs

# Function to train the GAN and save the generator model


def train_gan(GAN: GAN_Network, num_epochs, dataset: CustomDataset, save_dir, save_interval=10, lrg=0.0001, lrd=0.0001):
    lrg_ = str(lrg)[-4:]
    lrd_ = str(lrd)[-4:]
    GAN = GAN.to(device)
    GAN.train()
    GAN.generator.train()
    GAN.discriminator.train()
    d_criterion = nn.BCELoss()
    g_criterion = nn.BCEWithLogitsLoss()
    # gen_optimizer = optim.SGD(GAN.generator.parameters(), lr=0.001, momentum=0.9)
    # dis_optimizer = optim.SGD(GAN.discriminator.parameters(), lr=0.003, momentum=0.9)
    gen_optimizer = optim.Adam(GAN.parameters(), lr=lrg, betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(GAN.discriminator.parameters(), lr=lrd, betas=(0.5, 0.999))
    real_label = 1.
    fake_label = 0.
    if os.path.exists(save_dir + lrg_):
        # Remove all files in the directory and delete the directory
        for file in os.listdir(save_dir + lrg_):
            os.remove(os.path.join(save_dir + lrg_, file))
        os.rmdir(save_dir + lrg_)
    os.mkdir(save_dir + lrg_)
    if os.path.exists('disc_models' + lrd_):
        for file in os.listdir('disc_models' + lrd_):
            os.remove(os.path.join('disc_models' + lrd_, file))
        os.rmdir('disc_models' + lrd_)
    os.mkdir('disc_models' + lrd_)
    if os.path.exists('generated_images' + lrg_):
        for file in os.listdir('generated_images' + lrg_):
            os.remove(os.path.join('generated_images' + lrg_, file))
        os.rmdir('generated_images' + lrg_)
    os.mkdir('generated_images' + lrg_)
    fake_images = torch.randn((len(dataset), 1, 96, 128), dtype=torch.float32, device=device, requires_grad=True)
    for epoch in range(1, num_epochs + 1):
        half_batch_real = dataset.shuffle()
        half_batch_size = half_batch_real.size(0)
        half_batch_real = half_batch_real.to(device)
        half_batch_real_labels = torch.full((half_batch_size, 1), real_label, device=device)
        half_batch_fake = fake_images
        half_batch_fake_labels = torch.full((half_batch_size, 1), fake_label, device=device)

        d_train_X = torch.vstack((half_batch_real, half_batch_fake))
        # Train the discriminator
        dis_optimizer.zero_grad()
        d_out = GAN.discriminator(d_train_X)
        disc_loss_real = d_criterion(torch.sigmoid(d_out[:half_batch_size]), half_batch_real_labels)
        disc_loss_fake = d_criterion(torch.sigmoid(d_out[half_batch_size:]), half_batch_fake_labels)
        disc_loss = (disc_loss_real + disc_loss_fake) / 2
        disc_loss.backward()
        dis_optimizer.step()
        d_loss = disc_loss.detach().cpu().item()

        GAN.discriminator.eval()
        # Train the generator
        gen_optimizer.zero_grad()
        noise = normal_samples.sample([half_batch_size * 2]).to(device)  # type: ignore
        trick_labels = torch.full((half_batch_size * 2, 1), 1.0, device=device)
        gan_out = GAN(noise)
        gen_loss = g_criterion(gan_out, trick_labels)
        gen_loss.backward()
        gen_optimizer.step()

        g_loss = gen_loss.detach().cpu().item()

        GAN.discriminator.train()

        GAN.generator.eval()
        with torch.no_grad():
            fake_images = GAN.generator(normal_samples.sample([half_batch_size]).to(device))  # type: ignore
        GAN.generator.train()
        print(f"Epoch [{epoch}/{num_epochs}] Loss_D: {d_loss:.15f} Loss_G: {g_loss:.4f}")

        # Save the generator model at specified intervals
        if epoch % save_interval == 0:
            save_path_g = None
            if epoch % (save_interval * 2) == 0:

                save_path_g = os.path.join(save_dir + lrg_, f"generator_epoch_{epoch}.th")
                save_model(GAN.generator, save_path_g)

                save_path_d = os.path.join('disc_models' + lrd_, f"disc_epoch_{epoch}.th")
                save_model(GAN.discriminator, save_path_d)
                print(f"Models saved at epoch {epoch}.")

            if save_path_g is None:
                if len(os.listdir(save_dir + lrg_)) == 0:
                    save_model(GAN.generator, os.path.join(save_dir + lrg_, f"generator_epoch_{epoch}.th"))
                save_path_g = os.path.join(save_dir + lrg_, os.listdir(save_dir + lrg_)[-1])
            visualize_results(
                GAN.generator,
                num_samples=1,
                save_dir='generated_images' + lrg_ + '/' +
                str(epoch) +
                '.png')


# Function to visualize augmented training images and generator's output
noisy = normal_samples.sample([1]).to(device)  # type: ignore


def visualize_results(generator, num_samples=1, save_dir=None, noise=noisy):
    generator.eval()
    with torch.no_grad():
        generated_images = generator(noise).cpu().numpy()
    generator.train()

    for i in range(generated_images.shape[0]):
        generated_image = generated_images[i].squeeze(0)
        generated_image = (generated_image + 1) / 2
        if save_dir:
            plt.imsave(save_dir, generated_image, cmap='gray')

        else:
            plt.imshow(generated_image, cmap='gray')
            plt.axis("off")
            plt.show()


def show_data(data, n=6):
    for i in range(n):
        fig, ax = plt.subplots(figsize=(4, 4))
        img = data[i]
        img = img.numpy().transpose(1, 2, 0).squeeze()
        ax.imshow(img, cmap='gray')  # type: ignore
        plt.show()


def save_model(model, save_path):
    from torch import save
    return save(model.state_dict(), save_path)


def load_model(model_path='model.th'):
    from torch import load
    r = Generator(image_dim=(96, 128))
    r.load_state_dict(load(model_path, map_location='cpu'))
    return r


# Example usage with data augmentation
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-lrg", "--lr_gen", type=float, default=0.0001)
    parser.add_argument("-lrd", "--lr_dis", type=float, default=0.0003)
    args = parser.parse_args()
    lrg = args.lr_gen
    lrd = args.lr_dis
    image_dim = (96, 128)
    num_epochs = 10000
    save_dir = "generator_models"
    save_interval = 4
    GAN = GAN_Network()
    data_transform = transforms.Compose([
        transforms.RandomRotation(
            30,
            interpolation=transforms.InterpolationMode.NEAREST_EXACT),
        transforms.Lambda(lambda x: TF.adjust_contrast(x, 2)),
        transforms.Lambda(lambda x: TF.adjust_brightness(x, 1.25)),
        transforms.Lambda(lambda x: TF.adjust_sharpness(x, 20)),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((96, 128)),
        transforms.PILToTensor(),
    ])

    dataset = CustomDataset(root_dir="Hazel_Train", transform=data_transform)
    print(summary(GAN, (1, 1, 100), batch_dim=None))
    train_gan(GAN, num_epochs, dataset, save_dir, save_interval, lrg, lrd)
