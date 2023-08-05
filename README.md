# GAN Deep Learning

This repository contains Python scripts for training and visualizing a Generative Adversarial Network (GAN) for generating images. Generated images from the final model iteration are shown in the `eval_images.gif` file. 

## Requirements
- Python 3.6+
- PyTorch 1.7+
- TorchVision
- OpenCV
- Matplotlib
- ImageIO
- pillow

You can install the required packages using `pip`:
```bash
pip install torch torchvision opencv-python matplotlib imageio pillow
```

## CustomDataset

The `CustomDataset` class in `utils.py` provides a custom dataset for loading and transforming images. It allows you to load images from a specified directory and apply data augmentation and preprocessing. The `transform` parameter can be used to pass custom image transformations, but if it is not provided, a set of random transformations will be used.

## Models

The `Generator` and `Discriminator` models are implemented in `models.py`. The `Generator` is responsible for generating fake images, while the `Discriminator` is responsible for distinguishing between real and fake images.

## Training

To train the GAN, you can run the `train.py` script. You can specify various training parameters like the number of epochs, batch size, learning rates, etc., using command-line arguments.

Example usage:
```bash
python train.py --n_epochs 35000 --batch_size 81 --lrg 0.0026 --lrd 0.0008 --b1 0.5 --b2 0.999 --latent_dim 100 --img_size 96 128 --channels 1 --sample_interval 10 --dir images
```

This script will train the GAN for the specified number of epochs and save the generated images in the `images` directory. The loss values and learning rates are also saved in numpy files which can all be used later for debugging and visual validation.

## Visualizing the Model

To visualize the trained model, you can run the `train.py` script and passing the command line arguements `--v` and `--model`. This will then skip the training routine, load the model from the specified path and show generated images from the model.  

Example usage:
```bash
python train.py --model saved_models/generator9200.th --v 
```

This will generate and save evaluation images using the trained generator and create an MP4 video (`eval_imgs.mp4`) in the same directory. The video will be opened automatically after the process is complete.

## Note
Make sure to adjust the paths and other configurations in the scripts according to your specific use case.

Feel free to explore and modify the code to suit your needs! Happy GAN training!