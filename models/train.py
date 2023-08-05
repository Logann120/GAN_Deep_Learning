import argparse
from collections import deque
import os
import numpy as np
import torch
from torch import nn
from torchsummary import summary
from .models import Generator, Discriminator, device
from .utils import CustomDataset, make_mp4_from_imgs, plot_losses_and_save_plot, save_model, show_mp4, visualize_model
from torchvision.utils import save_image
normal_samples = torch.distributions.Normal(torch.full([100], 0.0), torch.full([100], 1.0))


@torch.no_grad()
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0., 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1., 0.02)
        nn.init.constant_(m.bias.data, 0.)
    else:
        pass


def compute_average_loss_at_each_epoch(loss_vals, window_size=10):
    avg_loss = []
    for i in range(window_size, len(loss_vals) + 1):
        avg_loss.append(np.mean(loss_vals[i - window_size:i]))
    return avg_loss


def train(n_epochs, batch_size, latent_dim, sample_interval, img_dir, lrg, lrd, b1, b2):

    # Loss function
    adversarial_loss = torch.nn.BCEWithLogitsLoss()
    # Initialize generator and discriminator
    generator = Generator().apply(weights_init)
    discriminator = Discriminator().apply(weights_init)
    print(summary(generator, (81, 100), batch_dim=None))
    print(summary(discriminator, (81, 1, 96 * 128), batch_dim=None))
    if torch.cuda.is_available():
        generator = generator.to(device)
        discriminator = discriminator.to(device)
        # adversarial_loss = adversarial_loss.to(device)

    dataset = CustomDataset(root_dir="Hazel_Train")
    # Optimizers
    # optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lrg)
    optimizer_G = torch.optim.NAdam(generator.parameters(), lr=lrg, betas=(b1, b2))
    # scheduler_G = torch.optim.lr_scheduler.CyclicLR(
    #    optimizer_G, base_lr=lrg / 3, max_lr=lrg, step_size_up=250, step_size_down=50, mode="triangular")
    # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lrd, betas=(b1, b2))
    # optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lrd)
    optimizer_D = torch.optim.NAdam(discriminator.parameters(), lr=lrd, betas=(b1, b2))
    # ----------
    #  Training
    # ----------
    d_losses, g_losses = [], []
    eval_inputs = normal_samples.sample(sample_shape=torch.Size([20])).to(device)
    # Create a stack to store the last 15 iterations of the generator model
    stack = deque()
    cooldown = 0
    for epoch in range(n_epochs):
        imgs = dataset.shuffle()
        # Adversarial ground truths
        valid = torch.full((imgs.size(0), 1), 1.0, dtype=torch.float32).to(device)
        fake = torch.full((imgs.size(0), 1), 0.0, dtype=torch.float32).to(device)

        # Configure input
        real_imgs = imgs.to(device)  # type: ignore

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_D.zero_grad()

        # Sample noise as generator input
        dz = normal_samples.sample(sample_shape=torch.Size([batch_size])).to(device)

        # Generate a batch of images
        fake_imgs = generator(dz).detach()

        # Loss measures generator's ability to fool the discriminator
        d_loss_fake = adversarial_loss(discriminator(fake_imgs), fake)
        d_loss_real = adversarial_loss(discriminator(real_imgs), valid)
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        gz = normal_samples.sample(sample_shape=torch.Size([batch_size * 2])).to(device)
        gen_imgs = generator(gz)
        trick_labels = torch.full((gen_imgs.size(0), 1), 1.0, dtype=torch.float32).to(device)
        g_loss = adversarial_loss(discriminator(gen_imgs), trick_labels)
        g_loss.backward()
        optimizer_G.step()
        # g_lr.append(optimizer_G.param_groups[0]["lr"])
        # scheduler_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        stack.append(generator.state_dict())
        if len(stack) > 15:
            stack.popleft()
        optimizer_D.zero_grad()

        dl = d_loss.detach().item()
        gl = g_loss.detach().item()
        d_losses.append(dl)
        g_losses.append(gl)

        if epoch % 10 == 0:
            print(
                "[Epoch %d/%d] [D loss: %.2f] [G loss: %.2f]" % (epoch, n_epochs, dl, gl))
            # generator.eval()
        if epoch % 4 == 0:
            generator.eval()
            eval_imgs = generator(eval_inputs)
            generator.train()
            save_image(((eval_imgs.cpu().detach() + 1) / 2).unsqueeze(1), img_dir +
                       "/%5d.png" % epoch, nrow=5, normalize=False)
        if dl >= 1.0 and cooldown == 0 and epoch > 2500:
            print('Current Epoch:', epoch)
            print('Saving Generators')
            # Save all the model instances stored in the stack to the disk
            j = 0
            while stack:
                state_dict = stack.pop()
                save_model(state_dict, os.path.join(
                    'saved_models', 'generator' + str(epoch - j) + '.th'))
                j += 1
            stack.clear()
            cooldown = 30
        if cooldown > 0:
            cooldown -= 1

    # d_loss_avgs = compute_average_loss_at_each_epoch(d_losses)
    # g_loss_avgs = compute_average_loss_at_each_epoch(g_losses)
    np.save('d_losses.npy', d_losses)
    np.save('g_losses.npy', g_losses)
    return d_losses, g_losses, None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=35000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=81, help="size of the batches")
    parser.add_argument("--lrg", type=float, default=0.0026, help="adam: learning rate")
    parser.add_argument("--lrd", type=float, default=0.0008, help="adam: learning rate")
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

    if not opt.v:
        try:
            os.mkdir(opt.dir)
        except BaseException:
            # Remove directory and all files
            for filename in os.listdir(opt.dir):
                os.remove(os.path.join(opt.dir, filename))
            os.rmdir(opt.dir)
            # Create directory
            os.makedirs(opt.dir, exist_ok=False)

        try:
            os.mkdir('saved_models')
        except BaseException:
            for filename in os.listdir('saved_models'):
                os.remove(os.path.join('saved_models', filename))
            os.rmdir('saved_models')
            os.makedirs('saved_models', exist_ok=False)

        d_losses, g_losses, d_lr, g_lr = train(opt.n_epochs, opt.batch_size, opt.latent_dim,
                                               opt.sample_interval, opt.dir, opt.lrg, opt.lrd, opt.b1, opt.b2)

        plot_losses_and_save_plot(d_losses, g_losses)
        # plot_lrs_and_save_plot(d_lr, g_lr)
        # plot_loss_with_lrs(d_losses, g_losses, d_lr, g_lr)
        # make_gif_from_imgs(opt.dir)
        # show_gif('movie.gif')

        make_mp4_from_imgs(opt.dir, 'movie.mp4', n=4)
        show_mp4('movie.mp4')

        """
        dataset = CustomDataset('Hazel_Train')
        for img in dataset.shuffle()[:16]:
            img = broken_glass(img)
            plt.imshow(img, cmap='gray')
            plt.show()
        """
    else:
        model_path = opt.model
        visualize_model(model_path, 100)
