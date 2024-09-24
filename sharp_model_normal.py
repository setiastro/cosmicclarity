import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
#radius_2_dir = r'C:\Users\Gaming\Desktop\Python Code\data\blurred_radius_2'
#radius_4_dir = r'C:\Users\Gaming\Desktop\Python Code\data\blurred_radius_4'
clean_dir = r'C:\Users\Gaming\Desktop\Python Code\data\clean_images'

# Create Datasets and DataLoaders for each blur radius
train_dataset_1 = SharpeningDataset(radius_1_dir, clean_dir)
train_loader_1 = DataLoader(train_dataset_1, batch_size=12, shuffle=True)

#train_dataset_2 = SharpeningDataset(radius_2_dir, clean_dir)
#train_loader_2 = DataLoader(train_dataset_2, batch_size=12, shuffle=True)

#train_dataset_4 = SharpeningDataset(radius_4_dir, clean_dir)
#train_loader_4 = DataLoader(train_dataset_4, batch_size=12, shuffle=True)

# Define the SharpeningCNN model with an additional convolutional layer
class SharpeningCNN(nn.Module):
    def __init__(self):
        super(SharpeningCNN, self).__init__()
        
        # Encoder (down-sampling path)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 1st convolutional layer (3 input channels -> 64 feature maps)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 2nd convolutional layer (64 -> 128 feature maps)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 3rd convolutional layer (128 -> 256 feature maps)
            nn.ReLU()
        )
        
        # Decoder (up-sampling path)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 1st deconvolutional layer (256 -> 128 feature maps)
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 2nd deconvolutional layer (128 -> 64 feature maps)
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # Output layer (64 -> 3 channels for RGB output)
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

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
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

            # Optionally, print the batch loss if you want more granular feedback
            #print(f'Batch [{batch_idx+1}/{num_batches}], Loss: {loss.item():.4f}')
        
        # Average loss over all batches
        avg_epoch_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Batch Loss: {avg_epoch_loss:.6f}')

    # Save the model
    torch.save(model.state_dict(), model_save_path)


# Instantiate three separate models for each blur radius
model_radius_1 = SharpeningCNN()
#model_radius_2 = SharpeningCNN()
#model_radius_4 = SharpeningCNN()

# Train each model with different epochs
print("Training model for blur radius 1 (10 epochs)...")
train_model(model_radius_1, train_loader_1, 10, 'sharp_cnn_radius_1.pth')

#print("Training model for blur radius 2 (12 epochs)...")
#train_model(model_radius_2, train_loader_2, 12, 'sharp_cnn_radius_2.pth')

#print("Training model for blur radius 4 (18 epochs)...")
#train_model(model_radius_4, train_loader_4, 18, 'sharp_cnn_radius_4.pth')
