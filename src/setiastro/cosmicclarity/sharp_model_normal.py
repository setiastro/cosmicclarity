import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# Define the PerceptualLoss using a pre-trained VGG model
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1').features[:16].eval()  # Use the first few layers of VGG16
        self.features = vgg
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        # Ensure the model is on the same device as the input
        device = pred.device
        self.features = self.features.to(device)
        
        pred_features = self.features(pred)
        target_features = self.features(target)
        loss = nn.MSELoss()(pred_features, target_features)
        return loss

# Define the combined L1 and Perceptual loss
class StellarCombinedLoss(nn.Module):
    def __init__(self):
        super(StellarCombinedLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        perceptual_loss = self.perceptual(pred, target)
        # Use a combination: 60% L1 and 40% Perceptual loss
        return 0.6 * l1_loss + 0.4 * perceptual_loss

# Define the SharpeningDataset class for a single blur radius
class SharpeningDataset(Dataset):
    def __init__(self, blur_dir, clean_dir, transform=None):
        self.blur_dir = blur_dir
        self.clean_dir = clean_dir
        self.image_names = os.listdir(blur_dir)  # Assuming the blur and clean dirs have the same image names
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        # Load the blurred and clean images
        blurred_img = cv2.imread(os.path.join(self.blur_dir, image_name))
        clean_img = cv2.imread(os.path.join(self.clean_dir, image_name))

        # Convert both images to float32 and normalize to [0, 1]
        blurred_img = blurred_img.astype(np.float32) / 255.0
        clean_img = clean_img.astype(np.float32) / 255.0

        # Convert images to (C, H, W) format
        blurred_img = np.transpose(blurred_img, (2, 0, 1))
        clean_img = np.transpose(clean_img, (2, 0, 1))

        blurred_tensor = torch.tensor(blurred_img)
        clean_tensor = torch.tensor(clean_img)

        if self.transform:
            blurred_tensor = self.transform(blurred_tensor)
            clean_tensor = self.transform(clean_tensor)

        return blurred_tensor, clean_tensor

# Define the paths to your dataset for each blur level
radius_1_dir = r'C:\Users\Gaming\Desktop\Python Code\data\blurred_radius_1'
clean_dir = r'C:\Users\Gaming\Desktop\Python Code\data\clean_images'

# Create Datasets and DataLoaders for each blur radius
train_dataset_1 = SharpeningDataset(radius_1_dir, clean_dir)
train_loader_1 = DataLoader(train_dataset_1, batch_size=12, shuffle=True)

# Define the SharpeningCNN model with adjusted convolutional layers
class SharpeningCNN(nn.Module):
    def __init__(self):
        super(SharpeningCNN, self).__init__()
        
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

    # Define the hybrid loss function (L1 + Perceptual Loss)
    criterion = StellarCombinedLoss().to(device)  # Ensure loss function is on the same device
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0  # Accumulating the total loss for all batches
        num_batches = len(train_loader)  # Total number of batches

        for batch_idx, (blurred_images, clean_images) in enumerate(train_loader):
            blurred_images, clean_images = blurred_images.to(device), clean_images.to(device)
            
            # Forward pass: generate sharpened images
            outputs = model(blurred_images)
            
            # Compute the loss
            loss = criterion(outputs, clean_images)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # Average loss over all batches
        avg_epoch_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Batch Loss: {avg_epoch_loss:.6f}')

    # Save the model
    torch.save(model.state_dict(), model_save_path)


def main():
    # Instantiate the model for blur radius 1
    model_radius_1 = SharpeningCNN()

    # Train the model for 10 epochs
    print("Training model for blur radius 1 (10 epochs)...")
    train_model(model_radius_1, train_loader_1, 10, 'sharp_cnn_radius_1.pth')


if __name__ == "__main__":
    main()
