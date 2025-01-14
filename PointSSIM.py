#PointSSIM generation
import numpy as np
from scipy.ndimage import distance_transform_edt, label, generate_binary_structure
from scipy.spatial import KDTree

def minimum_distance_transform(image):
    # The distance_transform_edt function computes the Euclidean distance to the nearest zero (background pixel)
    # Here, we invert the image because distance_transform_edt expects the background to be zero
    dt_image = distance_transform_edt(image)
    return dt_image


def local_maxima(image):
    # Pad the image with minimum possible value on edges to handle boundary conditions
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=np.min(image) - 1)

    # Create shifted versions for all eight neighbors
    center = padded_image[1:-1, 1:-1]
    top_left = padded_image[:-2, :-2]
    top = padded_image[:-2, 1:-1]
    top_right = padded_image[:-2, 2:]
    left = padded_image[1:-1, :-2]
    right = padded_image[1:-1, 2:]
    bottom_left = padded_image[2:, :-2]
    bottom = padded_image[2:, 1:-1]
    bottom_right = padded_image[2:, 2:]

    # Compare the center pixel to all its neighbors
    local_max = ((center >= top_left) & (center >= top) & (center >= top_right) &
                 (center >= left) & (center >= right) &
                 (center >= bottom_left) & (center >= bottom) & (center >= bottom_right))

    # Create the output array of the same shape as the original image
    output = np.zeros_like(image, dtype=bool)
    output[image > 0] = local_max[image > 0]

    return output


def calculate_statistics(counts):
    mean_count = np.mean(counts)
    var_count = np.var(counts)
    return mean_count, var_count

def count_ones_in_subgrids(binary_array, subgrid_ratio=0.10):
    rows, cols = binary_array.shape
    total_subgrid_row = int(subgrid_ratio * rows)
    total_subgrid_col = int(subgrid_ratio * cols)
    counts = []
    for i in range(0, rows, total_subgrid_row):
        for j in range(0, cols, total_subgrid_col):
            subgrid = binary_array[i:i+total_subgrid_row, j:j+total_subgrid_col]
            counts.append(np.sum(subgrid))
    return np.array(counts), total_subgrid_row, total_subgrid_col

def construct_metric(binary_array, subgrid_ratio=0.10):
    counts, total_subgrid_row, total_subgrid_col = count_ones_in_subgrids(binary_array, subgrid_ratio)
    mean_count, var_count = calculate_statistics(counts)
    
    # Expected mean and variance for a random distribution
    total_cells = total_subgrid_row * total_subgrid_col
    p = np.mean(binary_array)  # Proportion of 1s in the entire array
    expected_mean = total_cells * p
    expected_variance = total_cells * p * (1 - p)
    
    # Normalize the variance to be between -1 and 1
    metric = (var_count - expected_variance) / (var_count + expected_variance)
    # normalize the variance again to be between 0 and 1
    metric = (metric + 1) / 2
    
    return metric

def gridposition_to_coordinate(grid_position, grid_size, image_size = 1000):
    # Compute the size of the grid cells
    grid_cell_size = image_size / grid_size

    # Compute the x and y coordinates of the grid cell
    x = grid_position[0] * grid_cell_size 
    y = grid_position[1] * grid_cell_size 

    return x, y

def custom_distance_transform(array, image_size=1000):
    # Compute the Euclidean distance transform on the grid
    grid_distances = distance_transform_edt(array)
    
    # Convert grid distances to real-world coordinates by scaling
    grid_size = array.shape[0]
    grid_cell_size = image_size / grid_size
    real_distances = grid_distances * grid_cell_size
    
    return real_distances

def custom_anchors(image,base_size,arrays=False):
    mindist = minimum_distance_transform(image)
    locmin = local_maxima(mindist)
    anchors = np.zeros(image.shape)
    anchor_radii = np.zeros(image.shape)
    anchor_label = np.zeros(image.shape)
    indices = np.where(image == 1)
    coordinates = list(zip(indices[0], indices[1]))
    distances = mindist[indices]
    connectivity = generate_binary_structure(2, 2)
    labeled_array, num_features = label(image, structure=connectivity)

    # Sort coordinates by distances in descending order
    sorted_coords = [coord for _, coord in sorted(zip(distances, coordinates), reverse=True, key=lambda x: x[0])]
    # Initialize KD-Tree with the first point
    anchors[sorted_coords[0]] = 1
    anchor_radii[sorted_coords[0]] = mindist[sorted_coords[0]]
    anchor_label[sorted_coords[0]] = labeled_array[sorted_coords[0]]
    anchor_tree = KDTree([sorted_coords[0]])

    # Iterate over sorted coordinates
    for i in range(1, len(sorted_coords)):
        point = sorted_coords[i]
        if locmin[point] == 1:  # Check if the point is a local minimum
            distance, _ = anchor_tree.query(point)
            if distance >= mindist[point]:  # Compare distance to the nearest anchor
                anchors[point] = 1
                anchor_radii[point] = mindist[point]
                anchor_label[point] = labeled_array[point]
                anchor_tree = KDTree(np.vstack([anchor_tree.data, point]))  # Update KD-Tree with new anchor
    
    anchor_coordinates = np.argwhere(anchors)
    anchor_coord = np.array([np.flip(np.array(gridposition_to_coordinate(choord, image.shape[0],base_size))) for choord in anchor_coordinates])
    if arrays:
        return anchors, anchor_radii, anchor_label
    else:
        return anchor_coord, (anchor_radii[anchor_radii!=0]*base_size/image.shape[0]), anchor_label[anchor_label!=0]

def supervector(image):
    supervector = np.zeros(4)
    anchors, radii, labels = custom_anchors(image, 1000, arrays=True)
    # First feature is the sum of the radius of the anchor points divided by the number of grid cells
    supervector[0] = np.sum(radii**2) / (image.shape[0]*image.shape[1])
    # Second is the number of objects divided by the number of grid cells
    nr_of_objects = int(np.max(labels))
    supervector[1] = nr_of_objects 
    # Third is the number of anchor points per object divided by the total area of the object
    contribution = np.zeros(nr_of_objects)
    for i in range(nr_of_objects):
        object_cells = np.where(labels == i+1)
        nr_of_anchors_in_object = len(object_cells[0])
        sum_of_radii = np.sum(radii[object_cells]**2) / np.sum(radii**2)
        contribution[i] = nr_of_anchors_in_object 
    supervector[2] = np.mean(contribution)
    # Fourth is the custom metric
    supervector[3] = construct_metric(anchors)
    return supervector

def compute_metric(coordinates, k, base_size=1000):
    # Number of points
    n_points = len(coordinates)

    # Calculate total area of the grid
    total_area = base_size**2

    lambda_ = n_points / total_area

    # Compute the search radius as a multiple of the expected nearest neighbor distance
    search_radius = np.sqrt(k/(np.pi*lambda_))

    # Build KDTree for fast radius-based searches
    tree = KDTree(coordinates)

    # Get the number of points within the search radius for each point
    counts = []
    for point in coordinates:
        indices = tree.query_ball_point(point, search_radius)
        count = len(indices) - 1  # Exclude the point itself
        counts.append(count)

    # Compute the empirical mean and variance of the counts
    counts = np.array(counts)
    empirical_variance = np.var(counts)

    # Theoretical variance for a Poisson distribution (equals the mean)
    theoretical_variance = k

    # Compute the metric (empirical variance - theoretical variance) / (empirical variance + theoretical variance)
    if empirical_variance + theoretical_variance > 0:
        metric = (empirical_variance - theoretical_variance) / (empirical_variance + theoretical_variance)
    else:
        metric = 0

    return metric + 1

def disjoint_subregion_variance_measure(coordinates,base_size,subgrid_size=10):
    number_of_points = len(coordinates)
    theoretical_variance = number_of_points / subgrid_size**2
    # Count the number of points in each subregion
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

def feature_vector(image, base_size):
    supervector = np.zeros(4)
    anchors, radii, labels = custom_anchors(image, base_size)
    # Set k to be 1/10th of the anchor length
    k = 5
    # First feature is the number of anchor points
    supervector[0] = len(anchors)
    # Second is the sum of the radius of the anchor points divided by the base size area
    supervector[1] = np.sum(radii) / base_size**2
    #Third is the average number of anchor points per object
    nr_of_objects = int(np.max(labels))
    contribution = np.zeros(nr_of_objects)
    for i in range(nr_of_objects):
        object_cells = np.where(labels == i+1)
        nr_of_anchors_in_object = len(object_cells[0])
        contribution[i] = nr_of_anchors_in_object
    supervector[2] = np.mean(contribution)
    # Fourth is a custom metric
    supervector[3] = disjoint_subregion_variance_measure(anchors, base_size)
    return supervector

def point_SSIM(vec1,vec2):
    comp = np.zeros(4)
    for i in range(len(vec1-1)):
        comp[i] = ((vec1[i]-vec2[i])**2/np.max([vec1[i],vec2[i],vec1[i]-vec2[i]])**2)
    comp[3] = (vec1[i]-vec2[i])**2
    calc = np.mean(comp)
    PointSSIM = 1 - calc
    return PointSSIM