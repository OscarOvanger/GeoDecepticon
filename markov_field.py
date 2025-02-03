# Simple Markov field generator
# We generate binary 2D Markov field with the property that the value at any cell only depends on the 8-neighbourhood of that cell.
# We generate the field by sampling cell by cell row-wise, and for each cell we sample the value given the 8-neighbourhood of the cell.

import numpy as np

class MarkovImage2D:
    """
    A class to generate and evaluate 2D binary images from a Markov process
    defined by a prior, transition probabilities, and a local kernel.
    """
    def __init__(self, n, m,
                 prior_probs=(0.5, 0.5),
                 transition_matrix=None,
                 kernel_size=3,
                 sigma=1.0):
        """
        Parameters
        ----------
        n, m : int
            Dimensions of the 2D field.
        prior_probs : tuple or list of length 2
            (p0, p1) = Probability that a cell is 0 or 1 in the absence
            of neighbors (the prior).
        transition_matrix : 2x2 numpy array or list of lists
            transition_matrix[a][b] = P(new=b | old=a).
            For example, transition_matrix = [[0.9, 0.1],
                                              [0.2, 0.8]] means
            if old=0 => 90% chance new=0, 10% chance new=1.
        kernel_size : int
            Size of the local window (e.g., 3 => 3x3).
        sigma : float
            The Gaussian parameter in the kernel weighting.
        """
        self.n = n
        self.m = m
        self.prior_probs = prior_probs
        
        if transition_matrix is None:
            # Default to a simple random matrix, must be 2x2
            self.transition_matrix = np.array([[0.9, 0.1],
                                               [0.2, 0.8]])
        else:
            self.transition_matrix = np.array(transition_matrix, dtype=float)
        
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        # For convenience:
        self._half_k = kernel_size // 2  # e.g. 3//2 = 1

    def _kernel(self, i1, j1, i2, j2):
        """
        Gaussian kernel centered at (i1, j1), evaluated at (i2, j2).
        
        K((i1,j1),(i2,j2)) = exp( -|| (i1,j1)-(i2,j2) ||^2 / (2*sigma^2) )
        """
        dist_sq = (i1 - i2)**2 + (j1 - j2)**2
        return np.exp(-dist_sq / (2.0 * self.sigma**2))

    def _get_neighbors(self, i, j):
        """
        Return the list of neighbor coordinates within kernel_size x kernel_size
        that come BEFORE (i,j) in row-major order.
        We assume row-major ordering: (0,0), (0,1), ..., (0,m-1),
        (1,0), (1,1), ..., (n-1,m-1).
        """
        neighbors = []
        # define row/col bounds
        row_min = max(0, i - self._half_k)
        row_max = min(self.n - 1, i + self._half_k)
        col_min = max(0, j - self._half_k)
        col_max = min(self.m - 1, j + self._half_k)
        
        for r in range(row_min, row_max + 1):
            for c in range(col_min, col_max + 1):
                # (r,c) must come before (i,j) in generation order
                if (r < i) or (r == i and c < j):
                    neighbors.append((r, c))
        return neighbors

    def _compute_conditional_prob(self, field, i, j):
        """
        Compute P(x_ij = 0) and P(x_ij = 1) given the already-known neighbors
        in 'field'.

        field[i,j] might not be set yet (if called during generation).
        This function returns a tuple (p0, p1) that sums to 1.
        """
        # If (i,j) is the very first cell, use prior.
        if i == 0 and j == 0:
            return np.array(self.prior_probs, dtype=float)
        
        neighbors = self._get_neighbors(i, j)
        
        # If no neighbors found (should not happen except i=j=0), fallback to prior
        if len(neighbors) == 0:
            return np.array(self.prior_probs, dtype=float)
        
        # Weighted sum of transition probabilities based on neighbor values
        # We'll compute unnormalized p0 and p1; then normalize.
        p0_unnorm = 0.0
        p1_unnorm = 0.0
        
        # Each neighbor (r,c) has field[r,c] in {0,1}.
        # We get kernel weight * transition probability from that neighbor's value.
        # Then accumulate for x_ij=0 or x_ij=1.
        for (r, c) in neighbors:
            neighbor_value = field[r, c]
            # kernel weight
            w = self._kernel(i, j, r, c)
            
            # If neighbor is 0 => use row=0 from transition_matrix
            # If neighbor is 1 => use row=1 from transition_matrix
            # transition_matrix[neighbor_value][0] = P(new=0 | old=neighbor_value)
            # transition_matrix[neighbor_value][1] = P(new=1 | old=neighbor_value)
            p0_unnorm += w * self.transition_matrix[neighbor_value][0]
            p1_unnorm += w * self.transition_matrix[neighbor_value][1]
        
        # Normalize
        total = p0_unnorm + p1_unnorm
        if total == 0.0:
            # fallback if something degenerate: just use prior
            return np.array(self.prior_probs, dtype=float)
        
        return np.array([p0_unnorm / total, p1_unnorm / total])

    def generate_field(self, random_state=None):
        """
        Generate the entire n x m field of 0/1 using
        the Markov model in row-major (raster-scan) order.

        Returns
        -------
        field : numpy.ndarray
            A (n x m) array of 0/1 values.
        """
        if random_state is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(random_state)
        
        field = np.zeros((self.n, self.m), dtype=int)
        
        for i in range(self.n):
            for j in range(self.m):
                probs = self._compute_conditional_prob(field, i, j)
                # Sample from [0,1] with probability = probs
                field[i, j] = rng.choice([0, 1], p=probs)
        
        return field

    def joint_probability(self, field):
        """
        Compute the joint probability P(field).
        This is the product of local conditional probabilities in the
        Markov factorization, i.e.
        P(x_11) * P(x_12 | x_11) * ... * P(x_nm | neighbors).
        
        Parameters
        ----------
        field : 2D numpy array of shape (n, m)
            A fixed binary (0/1) image.
        
        Returns
        -------
        float
            Joint probability P(field).
        """
        p = 1.0
        for i in range(self.n):
            for j in range(self.m):
                # local distribution given neighbors
                local_probs = self._compute_conditional_prob(field, i, j)
                # multiply by the probability of the actual observed value in field[i,j]
                value = field[i,j]
                p *= local_probs[value]
        return p

    def conditional_probability(self, field, i, j, value=None):
        """
        Compute P(x_ij = value | all other cells of field).
        
        In a strictly local Markov model, we typically only need the neighbors
        of (i, j). But in 2D grid MRFs, the exact conditioning on 'all other cells'
        can be more involved. Here, we'll do a naive approach:
        
        1) Temporarily "forget" the field[i,j].
        2) Compute the local conditional distribution for x_ij
           using the same neighbor-based rule used at generation time.
        3) If value is not None, return just P(x_ij = value).
           Otherwise, return the distribution [p0, p1].
        """
        # Save the current value
        current_val = field[i,j]
        
        # We "remove" the value at (i,j), but to keep code consistent,
        # we'll just set it to something invalid temporarily or to 0
        # because _compute_conditional_prob() only uses neighbors that come BEFORE (i,j).
        # So for correctness, we only rely on neighbors that come before (i,j).
        # If you want a truly correct MRF approach, you'd need a more involved
        # inference routine. This is just a local-neighbor approach.
        field[i,j] = 0  # or 1, or anything—_compute_conditional_prob uses only earlier neighbors

        local_probs = self._compute_conditional_prob(field, i, j)
        
        # Restore the value
        field[i,j] = current_val
        
        if value is None:
            return local_probs
        else:
            return local_probs[value]

    def marginal_probability(self, field, i, j):
        """
        Naive local approach to get P(x_ij = 0) and P(x_ij = 1).
        In a full 2D MRF, the true marginal requires summing over
        all other configurations. This function returns the local 
        neighbor-based probabilities (i.e. the same approach as if
        we were generating x_ij in generation order).
        
        Returns a tuple (p0, p1).
        """
        p0, p1 = self.conditional_probability(field, i, j, value=None)
        return (p0, p1)


# -------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    # Create a MarkovImage2D object
    import matplotlib.pyplot as plt
    n, m = 64, 64
    prior = (0.5, 0.5)
    transition = [[0.99, 0.01],
                  [0.01, 0.99]]
    kernel_size = 5
    sigma = 0.1

    model = MarkovImage2D(n, m,
                          prior_probs=prior,
                          transition_matrix=transition,
                          kernel_size=kernel_size,
                          sigma=sigma)

    # Generate a field
    generated_field = model.generate_field(random_state=42)
    print("Generated Field:\n", generated_field)
    plt.imshow(generated_field, cmap='gray')
    plt.show()

    # Compute the joint probability of that field
    jp = model.joint_probability(generated_field)
    print("Joint Probability of generated field =", jp)

    # Conditional probability of x_2,2 = 1 given the rest
    cp = model.conditional_probability(generated_field, 2, 2, value=1)
    print(f"Conditional Probability that x_(2,2) = 1 given the rest = {cp:.4f}")

    # Marginal probability of x_2,2
    mp0, mp1 = model.marginal_probability(generated_field, 2, 2)
    print(f"Local Marginal Probability x_(2,2) = 0 => {mp0:.4f}, x_(2,2) = 1 => {mp1:.4f}")


# -------------------------------------------------------------------
# Here is my own implementation
#--------------------------------------------------------------------

from scipy.special import kv, gamma

def cov_mat(ind,idx, sigma=1.0,length_scale=10, nu=1.5):
    dist = np.linalg.norm(ind-idx)
    return sigma**2 * (1 + dist**2 / (length_scale**2))**(-nu)

def transition_mat(row,col, vals = np.array([[0.9,0.1],[0.2,0.8]])):
    return vals[row-1,col-1]

def prior_prob(ind, val):
    return 0.5

def conditional_prob(X, ind):
    not_nan_indices = np.where(~np.isnan(X[:, i]))[0]
    n = len(not_nan_indices)
    p_0 = 0
    p_1 = 0
    for i in range(n):
        idx = not_nan_indices[i]
        p_0 += cov_mat(ind,idx) * transition_mat(X[idx], 0) + prior_prob(ind,0)
        p_1 += cov_mat(ind,idx) * transition_mat(X[idx], 1) + prior_prob(ind,1)
    p_0 = p_0 / (p_0 + p_1)
    p_1 = 1 - p_0
    return p_0, p_1
        
