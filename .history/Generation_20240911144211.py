import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
import os
import numpy as np
from PIL import Image

# WGAN-GP Constants
LATENT_DIM = 100  # Size of the noise vector
IMG_SIZE = 128  # Size of the generated images
CHANNELS = 3  # Number of image channels (e.g., 3 for RGB images)
BATCH_SIZE = 64
LAMBDA_GP = 10  # Gradient penalty lambda hyperparameter
N_CRITIC = 5  # Number of critic iterations per generator iteration
EPOCHS = 20000  # Total number of epochs
SAMPLE_INTERVAL = 100

# Input and output directories
input_dir = r"C:\Users\andre\Downloads\7Dataset-20240717T141942Z-001\CleanedDatasets\Acne"  # Replace with the path to your dataset
output_dir = r"C:\Users\andre\Downloads\7Dataset-20240717T141942Z-001\Syntheticdatasets\Acne"  # Replace with the path to save synthetic images

# Create output directory if it doesn't exist
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

# Dataset for loading images from input directory
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Load image file paths from the directory
        self.data = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path).convert('RGB')  # Open image and ensure it's in RGB mode
        if self.transform:
            img = self.transform(img)
        return img

# Load dataset from the input directory
dataset = CustomDataset(root_dir=input_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Generator and Critic models
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, CHANNELS * IMG_SIZE * IMG_SIZE),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), CHANNELS, IMG_SIZE, IMG_SIZE)
        return img

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(CHANNELS * IMG_SIZE * IMG_SIZE, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Function to compute gradient penalty for WGAN-GP
def compute_gradient_penalty(critic, real_samples, fake_samples):
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    # Get critic's output for the interpolated samples
    d_interpolates = critic(interpolates)

    # Compute gradients of the critic's output with respect to the interpolated samples
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size(), device=real_samples.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Flatten the gradients and compute their norm
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

# Initialize generator and critic
generator = Generator()
critic = Critic()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.9))
optimizer_C = torch.optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.9))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Training loop
for epoch in range(EPOCHS):
    for i, imgs in enumerate(dataloader):
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

        print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] [D loss: {c_loss.item()}] [G loss: {g_loss.item()}]")

# Generate and save final synthetic images after the last epoch
print("Generating final images after training...")
z = Tensor(np.random.normal(0, 1, (TOTAL_IMAGES_TO_GENERATE, LATENT_DIM)))
final_synthetic_imgs = generator(z)
for j in range(TOTAL_IMAGES_TO_GENERATE):
    save_image(final_synthetic_imgs.data[j], f"{output_dir}/synthetic_final_{j}.png", normalize=True)

print(f"Generated {TOTAL_IMAGES_TO_GENERATE} final synthetic images and saved them in {output_dir}.")
