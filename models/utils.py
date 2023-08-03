import re
import cv2
import os
import random
import concurrent.futures
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.utils import save_image


class CustomDataset():
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = os.listdir(root_dir)
        if transform is None:
            self.transform = self.get_transforms()
        else:
            self.transform = transform
        self.images = self._load_images()

    def _load_image(self, img_name):
        image = Image.open(img_name).convert('RGB').convert("L")
        image = TF.center_crop(TF.resize(image,  # type: ignore
                                         [3300,
                                          4400],
                                         interpolation=transforms.InterpolationMode.NEAREST),
                               output_size=[int(3300 * 0.7),
                                            int(4400 * 0.7)])
        image = self.transform(image)
        self.transform = self.get_transforms()
        return image

    def _load_images(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._load_image, os.path.join(self.root_dir, img_name))  # type: ignore
                       for img_name in self.image_files]
            images = [f.result() for f in futures]

        return images

    def __len__(self):
        return len(self.image_files)

    def shuffle(self):
        transformed_images = torch.stack(self.images)
        return transformed_images[torch.randperm(transformed_images.size(0))]

    def get_transforms(self):
        a = np.random.uniform(1.1, 1.7)
        b = np.random.uniform(0.9, 1.2)
        c = np.random.uniform(1.1, 5.0)
        data_transform = transforms.Compose([
            transforms.Lambda(lambda x: TF.adjust_contrast(x, a)),
            transforms.Lambda(lambda x: TF.adjust_brightness(x, b)),
            transforms.Lambda(lambda x: TF.adjust_sharpness(x, c)),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((96, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        return data_transform


def save_model(model, save_path):
    from .models import Generator, Discriminator

    from torch import save
    if isinstance(model, (Generator, Discriminator)):
        return save(model.state_dict(), save_path)
    return save(model, save_path)


def load_model(model_path='model.th'):
    from .models import Generator, Discriminator
    from torch import load
    if 'generator' in model_path:
        r = Generator()
    else:
        r = Discriminator()
    r.load_state_dict(load(os.path.abspath(model_path), map_location='cpu'))
    return r


normal_samples = torch.distributions.Normal(torch.full([100], 0.0), torch.full([100], 1.0))


def visualize_model(fname='model.th', latent_dim=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = load_model(fname).to(device)
    generator.eval()
    if not os.path.exists('eval_imgs'):
        os.mkdir('eval_imgs')
    for i in range(800):
        eval_inputs = normal_samples.sample(torch.Size([20])).to(device)
        eval_imgs = generator(eval_inputs)
        save_image(((eval_imgs + 1) / 2).unsqueeze(1), 'eval_imgs' +
                   "/%5d.png" % (i), nrow=5, normalize=False)
    make_mp4_from_imgs('eval_imgs', 'eval_imgs.mp4')
    show_mp4('eval_imgs.mp4')


def plot_losses_and_save_plot(d_losses, g_losses, save_path='losses.png'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(d_losses, label="D")
    plt.plot(g_losses, label="G")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.xticks(
        np.linspace(
            0,
            len(d_losses),
            len(d_losses) // 100 + 1).astype(int),
        np.linspace(
            0,
            len(d_losses),
            len(d_losses) // 100 + 1).astype(int),
        rotation=60)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_lrs_and_save_plot(d_lrs, g_lrs, save_path='lrs.png'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Learning Rates During Training")
    plt.plot(d_lrs, label="D")
    plt.plot(g_lrs, label="G")
    plt.xlabel("iterations")
    plt.ylabel("Learning Rate")
    plt.xticks(np.linspace(0, len(d_lrs), 10, dtype=np.int8), np.linspace(0, len(d_lrs), 10, dtype=np.int8))
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_loss_with_lrs(d_losses, g_losses, d_lrs, g_lrs, save_path='losses_with_lrs.png'):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('Loss')
    ax1.plot(d_losses, label="D")
    ax1.plot(g_losses, label="G")
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate')
    ax2.plot(d_lrs, label="D", color='red')
    ax2.plot(g_lrs, label="G", color='green')
    ax2.tick_params(axis='y')
    fig.tight_layout()
    plt.show()


def make_gif_from_imgs(imgs_path, duration=500, loop=1, fname='movie.gif'):
    import imageio
    import os
    images = []
    for filename in sorted(os.listdir(imgs_path)):
        images.append(imageio.imread(os.path.join(imgs_path, filename)))
    imageio.mimsave(fname, images, duration=duration, loop=loop)


def make_mp4_from_imgs(imgs_path, fn='movie.mp4', n=6):
    img_array = []
    size = (0, 0)
    filenames = sorted(os.listdir(imgs_path), key=lambda x: int(re.findall(r'\d+', x)[0]))

    for filename in filenames:
        f = os.path.join(imgs_path, filename)
        img = cv2.imread(f)
        height, width, _ = img.shape
        size = (width, height)
        img_array.append(img)
    out = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*'mp4v'), n, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def show_gif(gif_path):
    os.startfile(gif_path)


def show_mp4(mp4_path):
    os.startfile(mp4_path)


def broken_glass(img):
    import numpy as np
    import cv2

    c, h, w = img.shape
    glass = img.numpy().transpose(1, 2, 0)
    h_start = 0
    s = h - 10
    w_start = 0
    ss = w - 10
    for i in range(10, h, 10):
        for j in range(10, w, 10):
            if np.random.rand() > 0.3:
                glass[h_start:i, w_start:j, 0] = img[0, s:h, ss:w - j]
    return glass
