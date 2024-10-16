import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16
from torchvision import transforms

# Perceptual Loss using VGG16
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features[:16].eval()  # Using first 16 layers from VGG16
        self.vgg = vgg
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Normalize images to match VGG expectations
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, denoised, noisefree):
        denoised_vgg = self.vgg(self.transform(denoised))
        noisefree_vgg = self.vgg(self.transform(noisefree))
        loss = F.mse_loss(denoised_vgg, noisefree_vgg)
        return loss

# Define the new combined loss function with PSNR, SSIM, MSE, and Perceptual Loss
class NewCombinedLoss(nn.Module):
    def __init__(self):
        super(NewCombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.ssim = torchmetrics.functional.structural_similarity_index_measure
        self.psnr = torchmetrics.functional.peak_signal_noise_ratio
        self.perceptual_loss = PerceptualLoss()
        self.sobel_weight = 0.4  # Adjust the weight of the Sobel loss

    def sobel_filter(self, image):
        # Convert the RGB image to grayscale by averaging the channels
        if image.shape[1] == 3:  # Check if the input has 3 channels (RGB)
            image = torch.mean(image, dim=1, keepdim=True)  # Convert to grayscale by averaging across channels

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        sobel_x = sobel_x.to(image.device)
        sobel_y = sobel_y.to(image.device)

        grad_x = F.conv2d(image, sobel_x, padding=1)
        grad_y = F.conv2d(image, sobel_y, padding=1)

        # Use abs() to ensure no negative values in the sum of squares
        grad = torch.sqrt(torch.abs(grad_x ** 2 + grad_y ** 2) + 1e-6)

        return grad

    def forward(self, pred, target):
        # Ensure that both predictions and targets are on the same device
        device = pred.device

        # PSNR
        psnr_loss = self.psnr(pred, target)

        # SSIM Loss
        ssim_loss = 1 - self.ssim(pred, target)

        # MSE Loss
        mse_loss = self.mse(pred, target)

        # Perceptual Loss (VGG-based)
        perceptual_loss_value = self.perceptual_loss(pred, target)

        # Sobel Loss (edge detection)
        pred_edges = self.sobel_filter(pred)
        target_edges = self.sobel_filter(target)
        sobel_loss = F.mse_loss(pred_edges, target_edges)

        # Combine losses with different weights
        combined_loss = (
            0.3 * mse_loss +
            0.3 * ssim_loss +
            0.2 * perceptual_loss_value +
            self.sobel_weight * sobel_loss
        )

        # Check for NaN in the final combined loss
        if torch.isnan(combined_loss).any():
            print("NaN detected in combined loss!")
            combined_loss = torch.tensor(0.0, device=device)  # Set to zero if NaN detected

        # Return the combined loss and other scores for logging
        return {
            'Combined Loss': combined_loss,
            'PSNR': psnr_loss.item(),
            'SSIM': ssim_loss.item(),
            'MSE': mse_loss.item(),
            'Perceptual Loss': perceptual_loss_value.item(),
            'Sobel Loss': sobel_loss.item(),
        }


# Define the DenoiseDataset class for a single noise level
class DenoiseDataset(Dataset):
    def __init__(self, noisey_dir, noisefree_dir, transform=None):
        self.noisey_dir = noisey_dir
        self.noisefree_dir = noisefree_dir
        self.image_names = os.listdir(noisey_dir)  # Assuming the noisey and noisefree dirs have the same image names
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        # Load the noisey and noisefree images
        noisey_img = cv2.imread(os.path.join(self.noisey_dir, image_name))
        noisefree_img = cv2.imread(os.path.join(self.noisefree_dir, image_name))

        # Convert both images to float32 and normalize to [0, 1]
        noisey_img = noisey_img.astype(np.float32) / 255.0
        noisefree_img = noisefree_img.astype(np.float32) / 255.0

        # Convert images to (C, H, W) format
        noisey_img = np.transpose(noisey_img, (2, 0, 1))
        noisefree_img = np.transpose(noisefree_img, (2, 0, 1))

        noisey_tensor = torch.tensor(noisey_img)
        noisefree_tensor = torch.tensor(noisefree_img)

        if self.transform:
            noisey_tensor = self.transform(noisey_tensor)
            noisefree_tensor = self.transform(noisefree_tensor)

        return noisey_tensor, noisefree_tensor


# Define the paths to your dataset for each noise level
noisey_dir = r'C:\Users\Gaming\Desktop\Python Code\data\noisey_images'
noisefree_dir = r'C:\Users\Gaming\Desktop\Python Code\data\noisefree_images'

# Create Datasets and DataLoaders for each noise level
train_dataset_ns = DenoiseDataset(noisey_dir, noisefree_dir)
train_loader_ns = DataLoader(train_dataset_ns, batch_size=12, shuffle=True)


# Define the DenoiseCNN model with adjusted convolutional layers
class DenoiseCNN(nn.Module):
    def __init__(self):
        super(DenoiseCNN, self).__init__()
        
        # Encoder (down-sampling path)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 1st layer (3 -> 16 feature maps)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 2nd layer (16 -> 32 feature maps)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 3rd layer (32 -> 64 feature maps)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 4th layer (64 -> 128 feature maps)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 5th layer (128 -> 256 feature maps)
            nn.ReLU()
        )
        
        # Decoder (up-sampling path)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 1st deconvolutional layer (256 -> 128 feature maps)
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 2nd deconvolutional layer (128 -> 64 feature maps)
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 3rd deconvolutional layer (64 -> 32 feature maps)
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 4th deconvolutional layer (32 -> 16 feature maps)
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),  # Output layer (16 -> 3 channels for RGB output)
            nn.Sigmoid()  # Ensure output values are between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Function to train a model with the provided DataLoader
def train_model(model, train_loader, num_epochs, model_save_path):
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the combined loss function and optimizer
    criterion = NewCombinedLoss().to(device)  # Ensure loss function is on the same device
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for noisey_images, noisefree_images in train_loader:
            noisey_images, noisefree_images = noisey_images.to(device), noisefree_images.to(device)
            
            # Forward pass: generate denoised images
            outputs = model(noisey_images)
            
            # Compute the loss
            losses = criterion(outputs, noisefree_images)
            combined_loss = losses['Combined Loss']
            
            # Backward pass and optimize
            optimizer.zero_grad()
            combined_loss.backward()

            # Clip gradients to avoid explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            
            epoch_loss += combined_loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

    # Save the model
    torch.save(model.state_dict(), model_save_path)


def main():
    # Instantiate models for noisey level
    model_ns = DenoiseCNN()

    # Train the model with different epochs
    print("Training denoise model (20 epochs)...")
    train_model(model_ns, train_loader_ns, 20, 'denoise_cnn.pth')


if __name__ == "__main__":
    main()