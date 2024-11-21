import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import BinaryImageDataset, preprocess_image
from transformer import VisionTransformer
import wandb

# Hyperparameters
embed_dim = 4
num_heads = 2
feedforward_dim = 8
num_layers = 2
num_tokens = 16
max_patches = 32 * 32
dropout = 0.1
learning_rate = 1e-4
num_epochs = 10
batch_size = 4

# Load data
images = torch.load("data/binary_images.pt")  # Save data in this format
dataset = BinaryImageDataset(images)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(embed_dim, num_heads, feedforward_dim, num_layers, num_tokens, max_patches, dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize W&B
wandb.init(project="vision-transformer")
wandb.watch(model, log="all")

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images in dataloader:
        images = images.to(device)
        patch_indices = torch.cat([preprocess_image(img) for img in images])
        patch_indices = patch_indices.to(device)

        masked_patches, mask = mask_patches(patch_indices)
        masked_patches, mask = masked_patches.to(device), mask.to(device)

        logits = model(masked_patches, mask)
        loss = criterion(logits.view(-1, num_tokens), patch_indices.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    wandb.log({"epoch_loss": avg_loss})
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

torch.save(model.state_dict(), "vision_transformer.pth")
wandb.save("vision_transformer.pth")



#NBNBNB KJØR DENNE I MORGEN
from dataloader import BinaryImageDataset
from transformer import VisionTransformer
import torch
import torch.optim as optim
import torch.nn as nn

# Dummy data
batch_size = 4
images = (torch.rand(batch_size, 64, 64) > 0.5).float()
dataset = BinaryImageDataset(images)

# Model setup
embed_dim = 4
num_tokens = 16
model = VisionTransformer(embed_dim, num_heads=2, feedforward_dim=8, num_layers=2, num_tokens=num_tokens, max_patches=32 * 32)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Preprocess images
patch_indices = torch.cat([preprocess_image(img) for img in images])
masked_patches = patch_indices.clone()
mask = torch.rand(masked_patches.shape) < 0.9  # Mask 90% of patches
masked_patches[mask] = num_tokens  # Mask token index

# Forward pass
logits = model(masked_patches.unsqueeze(0).long(), mask.unsqueeze(0))  # Add batch dim
loss = criterion(logits.view(-1, num_tokens), patch_indices.view(-1))

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Single training step passed. Loss:", loss.item())



# After training loop in train.py
from sample import sequential_inpainting
test_image = dataset[0]  # Get a test image from the dataset
inpainted_image = sequential_inpainting(model, test_image, mask_ratio=0.9, device=device)

# Visualize or save the result
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.title("Masked Image")
plt.imshow(test_image, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Inpainted Image")
plt.imshow(inpainted_image, cmap="gray")
plt.show()