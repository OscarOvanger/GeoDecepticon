# Load necessary libraries if not already loaded:
import torch
import numpy as np

# Assume training_data is already defined with shape (N, H, W)
N, H, W = training_data.shape

# Set parameters.
patch_size = 2
vocab_cap = 2000
vocab, counts, mask_token = build_vocabulary(training_data, patch_size, cap_size=vocab_cap)
patch_dim = patch_size ** 2
num_patches = (H // patch_size) ** 2
num_heads = 2
num_layers = 2
ffn_dim = 256
emb_dim = 64

# Create the model and load weights.
PATH = "model_epoch_1000.pth"
model = StackedContextViT(vocab, mask_token, patch_dim, num_patches,
                           emb_dim, num_heads, num_layers, ffn_dim)
# Note: if your torch.load expects "weights_only", adjust accordingly.
model.load_state_dict(torch.load(PATH, map_location='cpu'))  # remove weights_only if not applicable
model.eval()

# Generate 100 images using both methods for comparison.
manhattan_images = []
manhattan_ll = []
# Optionally define some condition indices/values
# (You can modify these as needed; here we provide an example.)
condition_indices = np.array([876,3825,2122,2892,1556,2683,3667,1767,483,2351,
                              2000,3312,2953,289,2373,2720,872,2713,1206,1341,
                              3541,2226,3423,1904,2882,2540,1497,2524,264,1441])
condition_values = np.array([0,1,1,1,0,0,1,0,0,1,0,0,0,1,0,1,0,0,1,0,
                              1,1,1,0,1,0,1,1,0,1])

for i in range(1000):
    gen_img_1, ll1 = generate_image(model, patch_size, W, condition_indices, condition_values,generation_order="manhattan")
