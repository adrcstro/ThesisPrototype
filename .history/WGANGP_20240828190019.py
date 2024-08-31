import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
import os

# WGAN-GP Constants
LATENT_DIM = 100  # Size of the noise vector
IMG_SIZE = 128  # Size of the generated images
CHANNELS = 3  # Number of image channels (e.g., 3 for RGB images)
BATCH_SIZE = 64
LAMBDA_GP = 10  # Gradient penalty lambda hyperparameter
N_CRITIC = 5  # Number of critic iterations per generator iteration
EPOCHS = 20000
SAMPLE_INTERVAL = 100

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

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = IMG_SIZE // 4
        self.l1 = nn.Sequential(nn.Linear(LATENT_DIM, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, CHANNELS, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Critic (Discriminator)
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(CHANNELS, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 3, stride=2, padding=1)
        )

    def forward(self, img):
        validity = self.model(img)
        return validity.view(validity.size(0), -1)

# Initialize generator and critic
generator = Generator()
critic = Critic()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.9))
optimizer_C = torch.optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.9))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Gradient penalty function
def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = Tensor(real_samples.size(0), 1, 1, 1).uniform_(0, 1)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = Tensor(real_samples.size(0), 1).fill_(1.0)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Training
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

        # Save Images
        if epoch % SAMPLE_INTERVAL == 0:
            save_image(gen_imgs.data[:25], f"images/{epoch}.png", nrow=5, normalize=True)

        print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] [D loss: {c_loss.item()}] [G loss: {g_loss.item()}]")
