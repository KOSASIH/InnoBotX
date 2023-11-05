import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd.variable import Variable

# Define the generator network
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Define the training procedure
def train_gan(generator, discriminator, dataloader, num_epochs, batch_size, input_size, device):
    generator.to(device)
    discriminator.to(device)

    # Define loss function and optimizers
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)

    # Training loop
    for epoch in range(num_epochs):
        for i, real_images in enumerate(dataloader):
            # Prepare real and fake labels
            real_labels = Variable(torch.ones(batch_size, 1)).to(device)
            fake_labels = Variable(torch.zeros(batch_size, 1)).to(device)

            # Train discriminator with real images
            real_images = real_images.view(-1, input_size).to(device)
            d_optimizer.zero_grad()
            real_outputs = discriminator(real_images)
            d_real_loss = criterion(real_outputs, real_labels)
            d_real_loss.backward()

            # Train discriminator with fake images
            z = Variable(torch.randn(batch_size, input_size)).to(device)
            fake_images = generator(z)
            fake_outputs = discriminator(fake_images.detach())
            d_fake_loss = criterion(fake_outputs, fake_labels)
            d_fake_loss.backward()
            d_optimizer.step()

            # Train generator
            generator.zero_grad()
            fake_outputs = discriminator(fake_images)
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()

            # Print training progress
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], d_real_loss: {:.4f}, d_fake_loss: {:.4f}, g_loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, len(dataloader), d_real_loss.item(), d_fake_loss.item(), g_loss.item()))

# Set random seed for reproducibility
torch.manual_seed(42)

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
input_size = 100
hidden_size = 128
output_size = 784
num_epochs = 50
batch_size = 100

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize generator and discriminator
generator = Generator(input_size, hidden_size, output_size)
discriminator = Discriminator(output_size, hidden_size, 1)

# Train GAN model
train_gan(generator, discriminator, dataloader, num_epochs, batch_size, input_size, device)

# Generate images using the trained generator
num_images = 10
z = Variable(torch.randn(num_images, input_size)).to(device)
generated_images = generator(z)

# Convert generated images to numpy arrays
generated_images = generated_images.detach().cpu().numpy()
generated_images = generated_images.reshape(-1, 28, 28)

# Display the generated images
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, num_images, figsize=(20, 4))
for i, ax in enumerate(axes):
    ax.imshow(generated_images[i], cmap='gray')
    ax.axis('off')
    ax.set_title('Generated Image {}'.format(i+1))

plt.show()
