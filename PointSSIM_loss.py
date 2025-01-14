# This is the PointSSIM class implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt, label, generate_binary_structure
from scipy.spatial import KDTree

class PointSSIMLoss(nn.Module):
    def __init__(self, base_size=1000):
        super(PointSSIMLoss, self).__init__()
        self.base_size = base_size

    def minimum_distance_transform(self, image):
        dt_image = distance_transform_edt(image)
        return dt_image

    def local_maxima(self, image):
        padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=np.min(image) - 1)
        center = padded_image[1:-1, 1:-1]
        top_left = padded_image[:-2, :-2]
        top = padded_image[:-2, 1:-1]
        top_right = padded_image[:-2, 2:]
        left = padded_image[1:-1, :-2]
        right = padded_image[1:-1, 2:]
        bottom_left = padded_image[2:, :-2]
        bottom = padded_image[2:, 1:-1]
        bottom_right = padded_image[2:, 2:]
        local_max = ((center >= top_left) & (center >= top) & (center >= top_right) &
                     (center >= left) & (center >= right) &
                     (center >= bottom_left) & (center >= bottom) & (center >= bottom_right))
        output = np.zeros_like(image, dtype=bool)
        output[image > 0] = local_max[image > 0]
        return output

    def gridposition_to_coordinate(self, grid_position, grid_size, image_size=1000):
        grid_cell_size = image_size / grid_size
        x = grid_position[0] * grid_cell_size
        y = grid_position[1] * grid_cell_size
        return x, y

    def disjoint_subregion_variance_measure(self, coordinates, base_size, subgrid_size=10):
        number_of_points = len(coordinates)
        theoretical_variance = number_of_points / subgrid_size**2
        counts = np.zeros((subgrid_size, subgrid_size))
        for point in coordinates:
            x, y = point
            i = int(x // (base_size / subgrid_size))
            j = int(y // (base_size / subgrid_size))
            if i == subgrid_size:
                i -= 1
            if j == subgrid_size:
                j -= 1
            counts[i, j] += 1
        empirical_variance = np.var(counts)
        measure = 1 / (1 + theoretical_variance/empirical_variance)
        return measure

    def custom_anchors(self, image, base_size, arrays=False):
        mindist = self.minimum_distance_transform(image)
        locmin = self.local_maxima(mindist)
        anchors = np.zeros(image.shape)
        anchor_radii = np.zeros(image.shape)
        anchor_label = np.zeros(image.shape)
        connectivity = generate_binary_structure(2, 2)

        labeled_array, num_features = label(image, structure=connectivity)
        indices = np.where(image == 1)
        coordinates = list(zip(indices[0], indices[1]))
        distances = mindist[indices]

        sorted_coords = [coord for _, coord in sorted(zip(distances, coordinates), reverse=True, key=lambda x: x[0])]

        anchors[sorted_coords[0]] = 1
        anchor_radii[sorted_coords[0]] = mindist[sorted_coords[0]]
        anchor_label[sorted_coords[0]] = labeled_array[sorted_coords[0]]
        anchor_tree = KDTree([sorted_coords[0]])

        for i in range(1, len(sorted_coords)):
            point = sorted_coords[i]
            if locmin[point] == 1:
                distance, _ = anchor_tree.query(point)
                if distance >= mindist[point]:
                    anchors[point] = 1
                    anchor_radii[point] = mindist[point]
                    anchor_label[point] = labeled_array[point]
                    anchor_tree = KDTree(np.vstack([anchor_tree.data, point]))

        anchor_coordinates = np.argwhere(anchors)
        anchor_coord = np.array([np.flip(np.array(self.gridposition_to_coordinate(choord, image.shape[0], base_size))) 
                                 for choord in anchor_coordinates])
        if arrays:
            return anchors, anchor_radii, anchor_label
        else:
            return anchor_coord, (anchor_radii[anchor_radii!=0]*base_size/image.shape[0]), anchor_label[anchor_label!=0]

    def feature_vector(self, image, base_size):
        start_time = time.time()
        # image is numpy array (after conversion)
        anchors, radii, labels = self.custom_anchors(image, base_size)
        k = 5
        supervector = np.zeros(4)
        # First feature: number of anchor points
        supervector[0] = len(anchors)
        # Second: sum of radius of the anchor points divided by base_size^2
        supervector[1] = np.sum(radii) / (base_size**2)
        nr_of_objects = int(np.max(labels))
        contribution = np.zeros(nr_of_objects)
        for i in range(nr_of_objects):
            object_cells = np.where(labels == i+1)
            nr_of_anchors_in_object = len(object_cells[0])
            contribution[i] = nr_of_anchors_in_object
        supervector[2] = np.mean(contribution)
        # Fourth: a custom metric
        supervector[3] = self.disjoint_subregion_variance_measure(anchors, base_size)
        return supervector

    def point_SSIM(self, vec1, vec2):
        # vec1, vec2: torch tensors of shape (4,)
        # We'll replicate the logic from the given code, but ensure indexing correctness.
        # Original code seems to have a small indexing bug in the loop for comp.
        comp = torch.zeros(4, dtype=vec1.dtype, device=vec1.device)
        for i in range(3):
            denom = torch.max(torch.tensor([vec1[i], vec2[i], vec1[i]-vec2[i]], device=vec1.device))
            comp[i] = ((vec1[i]-vec2[i])**2) / (denom**2 if denom != 0 else 1e-8)
        # The last component is computed differently in the original code
        comp[3] = (vec1[3]-vec2[3])**2
        calc = torch.mean(comp)
        PointSSIM = 1 - calc
        return PointSSIM

    def forward(self, img_batch1, img_batch2):
        """
        img_batch1, img_batch2: PyTorch tensors of shape (N, H, W) or (N, 1, H, W)
        with values in {0,1} indicating binary images.
        """
        if img_batch1.dim() == 4:
            img_batch1 = img_batch1.squeeze(1)
        if img_batch2.dim() == 4:
            img_batch2 = img_batch2.squeeze(1)

        # Convert to numpy for processing (non-differentiable part)
        img_batch1_np = img_batch1.cpu().numpy()
        img_batch2_np = img_batch2.cpu().numpy()

        # Compute feature vectors for each image in the batch
        features1 = []
        features2 = []
        for i in range(img_batch1_np.shape[0]):
            f1 = self.feature_vector(img_batch1_np[i], self.base_size)
            f2 = self.feature_vector(img_batch2_np[i], self.base_size)
            features1.append(f1)
            features2.append(f2)
        
        # Convert features back to torch for differentiable operations at the end
        features1 = torch.tensor(features1, dtype=img_batch1.dtype, device=img_batch1.device)
        features2 = torch.tensor(features2, dtype=img_batch2.dtype, device=img_batch2.device)

        # Compute point_SSIM for each pair and then mean over batch
        scores = []
        for i in range(features1.shape[0]):
            score = self.point_SSIM(features1[i], features2[i])
            scores.append(score)
        scores = torch.stack(scores)
        return scores

    def backward(self, *args, **kwargs):
        # Since the feature extraction steps are non-differentiable, 
        # no meaningful gradients with respect to original images will be computed.
        # The default PyTorch autograd will handle gradients w.r.t. the final computations if possible.
        pass



class PointSSIMLoss_fast(nn.Module):
    def __init__(self, base_size=1000):
        super(PointSSIMLoss_fast, self).__init__()
        self.base_size = base_size

    def minimum_distance_transform(self, image):
        dt_image = distance_transform_edt(image)
        return dt_image

    def local_maxima(self, image):
        padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=np.min(image) - 1)
        center = padded_image[1:-1, 1:-1]
        top_left = padded_image[:-2, :-2]
        top = padded_image[:-2, 1:-1]
        top_right = padded_image[:-2, 2:]
        left = padded_image[1:-1, :-2]
        right = padded_image[1:-1, 2:]
        bottom_left = padded_image[2:, :-2]
        bottom = padded_image[2:, 1:-1]
        bottom_right = padded_image[2:, 2:]
        local_max = ((center >= top_left) & (center >= top) & (center >= top_right) &
                     (center >= left) & (center >= right) &
                     (center >= bottom_left) & (center >= bottom) & (center >= bottom_right))
        output = np.zeros_like(image, dtype=bool)
        output[image > 0] = local_max[image > 0]
        return output

    def gridposition_to_coordinate(self, grid_position, grid_size, image_size=1000):
        grid_cell_size = image_size / grid_size
        x = grid_position[0] * grid_cell_size
        y = grid_position[1] * grid_cell_size
        return x, y

    def disjoint_subregion_variance_measure(coordinates, base_size, subgrid_size=10):
        number_of_points = len(coordinates)
        if number_of_points == 0:
            # same fallback as before
            return 1.0

        theoretical_variance = number_of_points / (subgrid_size**2)
        
        # coordinates is Nx2 -> convert to "bin indices"
        coords = np.array(coordinates)
        cell_size = base_size / subgrid_size
        i = np.floor(coords[:,0] / cell_size).astype(np.int32)  # row
        j = np.floor(coords[:,1] / cell_size).astype(np.int32)  # col
        # clamp to subgrid_size-1
        i = np.clip(i, 0, subgrid_size-1)
        j = np.clip(j, 0, subgrid_size-1)
        
        # flatten bin index
        bin_index = i * subgrid_size + j
        # count how many points in each bin
        counts_1d = np.bincount(bin_index, minlength=subgrid_size*subgrid_size)
        counts_2d = counts_1d.reshape(subgrid_size, subgrid_size)
        
        empirical_variance = np.var(counts_2d)
        if empirical_variance < 1e-8:
            return 1.0
        
        measure = 1.0 / (1.0 + (theoretical_variance/empirical_variance))
        return measure

    def custom_anchors(self, image, base_size, arrays=False):
        mindist = self.minimum_distance_transform(image)
        locmin = self.local_maxima(mindist)
        anchors = np.zeros(image.shape)
        anchor_radii = np.zeros(image.shape)
        anchor_label = np.zeros(image.shape)
        connectivity = generate_binary_structure(2, 2)

        labeled_array, num_features = label(image, structure=connectivity)
        indices = np.where(image == 1)
        coordinates = list(zip(indices[0], indices[1]))
        distances = mindist[indices]

        sorted_coords = [coord for _, coord in sorted(zip(distances, coordinates), 
                                                      reverse=True, key=lambda x: x[0])]
        # Add first anchor
        anchors[sorted_coords[0]] = 1
        anchor_radii[sorted_coords[0]] = mindist[sorted_coords[0]]
        anchor_label[sorted_coords[0]] = labeled_array[sorted_coords[0]]
        anchor_tree = KDTree([sorted_coords[0]])

        for i in range(1, len(sorted_coords)):
            point = sorted_coords[i]
            if locmin[point] == 1:
                distance, _ = anchor_tree.query(point)
                if distance >= mindist[point]:
                    anchors[point] = 1
                    anchor_radii[point] = mindist[point]
                    anchor_label[point] = labeled_array[point]
                    anchor_tree = KDTree(np.vstack([anchor_tree.data, point]))

        anchor_coordinates = np.argwhere(anchors)
        anchor_coord = np.array([
            np.flip(np.array(self.gridposition_to_coordinate(choord, image.shape[0], base_size))) 
            for choord in anchor_coordinates
        ])

        if arrays:
            return anchors, anchor_radii, anchor_label
        else:
            return (anchor_coord, 
                    (anchor_radii[anchor_radii != 0] * base_size / image.shape[0]),
                    anchor_label[anchor_label != 0])

    def feature_vector(self, image, base_size):
        anchors, radii, labels = self.custom_anchors(image, base_size)
        supervector = np.zeros(4)
        supervector[0] = len(anchors)
        supervector[1] = np.sum(radii) / (base_size**2)
        nr_of_objects = int(np.max(labels)) if labels.size > 0 else 0
        contribution = np.zeros(nr_of_objects)
        for i in range(nr_of_objects):
            object_cells = np.where(labels == i+1)
            nr_of_anchors_in_object = len(object_cells[0])
            contribution[i] = nr_of_anchors_in_object
        supervector[2] = np.mean(contribution) if contribution.size > 0 else 0
        supervector[3] = self.disjoint_subregion_variance_measure(anchors, base_size)
        return supervector

    def point_SSIM(self, vec1, vec2):
        comp = torch.zeros(4, dtype=vec1.dtype, device=vec1.device)
        for i in range(3):
            denom = torch.max(torch.tensor([vec1[i], vec2[i], vec1[i]-vec2[i]], device=vec1.device))
            comp[i] = ((vec1[i] - vec2[i])**2) / (denom**2 if denom != 0 else 1e-8)
        comp[3] = (vec1[3] - vec2[3])**2
        calc = torch.mean(comp)
        PointSSIM = 1 - calc
        return PointSSIM

    def forward(self, img_batch1, img_batch2):
        """
        Sequential version for comparison.
        img_batch1, img_batch2: shape (N, H, W) or (N, 1, H, W)
        """
        if img_batch1.dim() == 4:
            img_batch1 = img_batch1.squeeze(1)
        if img_batch2.dim() == 4:
            img_batch2 = img_batch2.squeeze(1)

        # Convert to numpy
        img_batch1_np = img_batch1.cpu().numpy()
        img_batch2_np = img_batch2.cpu().numpy()

        # For each image in the batch, compute feature vectors
        features1 = []
        features2 = []
        for i in range(img_batch1_np.shape[0]):
            f1 = self.feature_vector(img_batch1_np[i], self.base_size)
            f2 = self.feature_vector(img_batch2_np[i], self.base_size)
            features1.append(f1)
            features2.append(f2)
        
        # Convert features to torch
        features1 = torch.tensor(features1, dtype=img_batch1.dtype, device=img_batch1.device)
        features2 = torch.tensor(features2, dtype=img_batch2.dtype, device=img_batch2.device)

        # Compute point_SSIM for each image pair
        scores = []
        for i in range(features1.shape[0]):
            score = self.point_SSIM(features1[i], features2[i])
            scores.append(score)
        scores = torch.stack(scores)
        return scores

    def backward(self, *args, **kwargs):
        # Non-differentiable steps => no gradients from feature extraction
        pass