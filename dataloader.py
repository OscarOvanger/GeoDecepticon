import torch
from torch.utils.data import Dataset

class BinaryImageDataset(Dataset):
    def __init__(self, images):
        """
        Args:
            images (Tensor): Tensor of shape (num_images, 64, 64) with binary values (0 or 1).
        """
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return torch.tensor(image, dtype=torch.float32)

def preprocess_image(image):
    """
    Splits a binary 64x64 image into flattened 2x2 patches.
    Returns patch indices tensor (num_patches, 4).
    """
    patches = image.unfold(0, 2, 2).unfold(1, 2, 2)
    patches = patches.contiguous().view(-1, 4)  # (num_patches, 4)
    patch_indices = (patches * torch.tensor([8, 4, 2, 1])).sum(dim=1)  # Convert binary patch to index
    return patch_indices

"""
# Create a dummy dataset
num_images = 5
images = (torch.rand(num_images, 64, 64) > 0.5).float()  # Random binary images
dataset = BinaryImageDataset(images)

print("Testing BinaryImageDataset...")
# Check the dataset length
assert len(dataset) == num_images, "Dataset length is incorrect"
print("Dataset length test passed.")

# Check if the first item is a 64x64 binary image
first_image = dataset[0]
assert first_image.shape == (64, 64), "Image shape is incorrect"
assert first_image.min() >= 0 and first_image.max() <= 1, "Image values should be binary (0 or 1)"
print("Dataset indexing and image format test passed.")

print("Testing preprocess_image...")
# Test preprocess_image function
test_image = (torch.rand(64, 64) > 0.5).float()  # Single random binary image
patch_indices = preprocess_image(test_image)

# Check the shape of the output
assert patch_indices.shape == (32 * 32,), "Preprocessed patch indices shape is incorrect"

# Check the range of indices
assert patch_indices.min() >= 0 and patch_indices.max() < 16, "Patch indices should be in range [0, 15]"
print("Preprocess image test passed.")

print("All tests passed!")
"""