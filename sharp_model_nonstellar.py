import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import Dataset, DataLoader

def sobel_filter(image):
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
    grad = torch.sqrt(torch.abs(grad_x**2 + grad_y**2) + 1e-6)
    
    return grad




# Define the combined loss function with individual NaN checks
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.ssim = torchmetrics.functional.structural_similarity_index_measure
        self.sobel_weight = 0.4  # Adjust the weight of the Sobel loss

    def forward(self, pred, target):
        # Ensure that both predictions and targets are on the same device
        device = pred.device

        # MSE Loss
        mse_loss = self.mse(pred, target)
        if torch.isnan(mse_loss).any():
            print("NaN detected in MSE loss!")
            mse_loss = torch.tensor(0.0, device=device)  # Set to zero if NaN detected

        # SSIM Loss
        ssim_loss = 1 - self.ssim(pred, target)
        if torch.isnan(ssim_loss).any():
            print("NaN detected in SSIM loss!")
            ssim_loss = torch.tensor(0.0, device=device)  # Set to zero if NaN detected

        # Sobel Loss (edge detection)
        pred_edges = sobel_filter(pred)
        target_edges = sobel_filter(target)
        sobel_loss = F.mse_loss(pred_edges, target_edges)
        if torch.isnan(sobel_loss).any():
            print("NaN detected in Sobel loss!")
            sobel_loss = torch.tensor(0.0, device=device)  # Set to zero if NaN detected

        # Combine losses with different weights
        combined_loss = 0.3 * mse_loss + 0.3 * ssim_loss + self.sobel_weight * sobel_loss

        # Check for NaN in the final combined loss
        if torch.isnan(combined_loss).any():
            print("NaN detected in combined loss!")
            combined_loss = torch.tensor(0.0, device=device)  # Set to zero if NaN detected

        return combined_loss


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
nonstellar_radius_1_dir = r'C:\Users\Gaming\Desktop\Python Code\data\nonstellar_blurred_radius_1'
nonstellar_radius_2_dir = r'C:\Users\Gaming\Desktop\Python Code\data\nonstellar_blurred_radius_2'
nonstellar_radius_4_dir = r'C:\Users\Gaming\Desktop\Python Code\data\nonstellar_blurred_radius_4'
nonstellar_radius_8_dir = r'C:\Users\Gaming\Desktop\Python Code\data\nonstellar_blurred_radius_8'
clean_nonstellar_dir = r'C:\Users\Gaming\Desktop\Python Code\data\nonstellar_clean_images'

# Create Datasets and DataLoaders for each blur radius
train_dataset_ns_1 = SharpeningDataset(nonstellar_radius_1_dir, clean_nonstellar_dir)
train_loader_ns_1 = DataLoader(train_dataset_ns_1, batch_size=12, shuffle=True)

train_dataset_ns_2 = SharpeningDataset(nonstellar_radius_2_dir, clean_nonstellar_dir)
train_loader_ns_2 = DataLoader(train_dataset_ns_2, batch_size=12, shuffle=True)

train_dataset_ns_4 = SharpeningDataset(nonstellar_radius_4_dir, clean_nonstellar_dir)
train_loader_ns_4 = DataLoader(train_dataset_ns_4, batch_size=12, shuffle=True)

train_dataset_ns_8 = SharpeningDataset(nonstellar_radius_8_dir, clean_nonstellar_dir)
train_loader_ns_8 = DataLoader(train_dataset_ns_8, batch_size=12, shuffle=True)

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

    # Define the combined loss function and optimizer
    criterion = CombinedLoss().to(device)  # Ensure loss function is on the same device
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for blurred_images, clean_images in train_loader:
            blurred_images, clean_images = blurred_images.to(device), clean_images.to(device)
            
            # Forward pass: generate sharpened images
            outputs = model(blurred_images)
            
            # Compute the loss
            loss = criterion(outputs, clean_images)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients to avoid explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

    # Save the model
    torch.save(model.state_dict(), model_save_path)

# Instantiate models for each blur radius
model_ns_radius_1 = SharpeningCNN()
model_ns_radius_2 = SharpeningCNN()
model_ns_radius_4 = SharpeningCNN()
model_ns_radius_8 = SharpeningCNN()

# Train each model with different epochs
print("Training non-stellar model for blur radius 1 (6 epochs)...")
train_model(model_ns_radius_1, train_loader_ns_1, 6, 'nonstellar_sharp_cnn_radius_1.pth')

print("Training non-stellar model for blur radius 2 (12 epochs)...")
train_model(model_ns_radius_2, train_loader_ns_2, 12, 'nonstellar_sharp_cnn_radius_2.pth')

print("Training non-stellar model for blur radius 4 (18 epochs)...")
train_model(model_ns_radius_4, train_loader_ns_4, 18, 'nonstellar_sharp_cnn_radius_4.pth')

print("Training non-stellar model for blur radius 8 (20 epochs)...")
train_model(model_ns_radius_8, train_loader_ns_8, 20, 'nonstellar_sharp_cnn_radius_8.pth')
