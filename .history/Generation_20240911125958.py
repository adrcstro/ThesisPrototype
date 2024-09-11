import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
import os
import numpy as np

# WGAN-GP Constants
LATENT_DIM = 100  # Size of the noise vector
IMG_SIZE = 128  # Size of the generated images
CHANNELS = 3  # Number of image channels (e.g., 3 for RGB images)
BATCH_SIZE = 64
LAMBDA_GP = 10  # Gradient penalty lambda hyperparameter
N_CRITIC = 5  # Number of critic iterations per generator iteration
EPOCHS = 20000
SAMPLE_INTERVAL = 100

# Create output directory for synthetic images
output_dir = "synthetic_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Total number of images to save
TOTAL_IMAGES_TO_GENERATE = 100
generated_image_count = 0

# Image transformation
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * CHANNELS, [0.5] * CHANNELS)
])

# Dataset placeholder (replace this with your actual dataset)
class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        # Load your dataset here (e.g., image file paths)
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform:
            img = self.transform(img)
        return img

dataset = CustomDataset(transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Generator and Critic (same as before)

# Initialize generator and critic
generator = Generator()
critic = Critic()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.9))
optimizer_C = torch.optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.9))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Gradient penalty function (same as before)

# Training loop
for epoch in range(EPOCHS):
    for i, imgs in enumerate(dataloader):
        if generated_image_count >= TOTAL_IMAGES_TO_GENERATE:
            print(f"Generated {generated_image_count} images. Stopping training.")
            break
        
        # Configure input
        real_imgs = imgs.type(Tensor)
        
        # Train Critic
        optimizer_C.zero_grad()
        
        z = Tensor(np.random.normal(0, 1, (imgs.size(0), LATENT_DIM)))
        fake_imgs = generator(z).detach()
        real_validity = critic(real_imgs)
        fake_validity = critic(fake_imgs)
        gradient_penalty = compute_gradient_penalty(critic, real_imgs.data, fake_imgs.data)
        c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + LAMBDA_GP * gradient_penalty
        
        c_loss.backward()
        optimizer_C.step()
        
        # Train Generator every N_CRITIC steps
        if i % N_CRITIC == 0:
            optimizer_G.zero_grad()
            
            gen_imgs = generator(z)
            g_loss = -torch.mean(critic(gen_imgs))
            
            g_loss.backward()
            optimizer_G.step()

            # Save synthetic images
            for j in range(gen_imgs.size(0)):
                if generated_image_count >= TOTAL_IMAGES_TO_GENERATE:
                    break
                save_image(gen_imgs.data[j], f"{output_dir}/synthetic_{generated_image_count}.png", normalize=True)
                generated_image_count += 1

        print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] [D loss: {c_loss.item()}] [G loss: {g_loss.item()}]")

    if generated_image_count >= TOTAL_IMAGES_TO_GENERATE:
        break
