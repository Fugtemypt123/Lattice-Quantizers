import numpy as np

# fix random seed for reproducibility
np.random.seed(0)

# Function to generate a random n x m matrix with Gaussian entries
def GRAN(n, m):
    return np.random.normal(0, 1, (n, m))

# Function to generate a random vector uniformly distributed in [0, 1)
def URAN(n):
    return np.random.uniform(0, 1, n)

# Function to find the closest lattice point
def CLP(B, zB):
    # Approximation of the closest point in the lattice (solving lattice problem is hard)
    return np.round(np.linalg.solve(B, zB))


def gram_schmidt(B):
    """Perform Gram-Schmidt orthogonalization."""
    n, m = B.shape
    R = np.zeros((n, n))
    Q = np.zeros_like(B)

    for i in range(n):
        Q[i] = B[i]
        for j in range(i):
            R[j, i] = np.dot(Q[j], B[i]) / np.dot(Q[j], Q[j])
            Q[i] -= R[j, i] * Q[j]
        R[i, i] = np.linalg.norm(Q[i])
        Q[i] = Q[i] / R[i, i]
    
    return Q, R

def lll_reduction(B, delta=0.75):
    """LLL (Lenstra–Lenstra–Lovász) lattice basis reduction."""
    n = B.shape[0]
    B = B.copy()
    Q, R = gram_schmidt(B)  # Initial Gram-Schmidt orthogonalization
    
    k = 1
    while k < n:
        for j in range(k-1, -1, -1):
            # Reduce the basis vectors
            mu = np.round(np.dot(B[k], Q[j]) / np.dot(Q[j], Q[j]))
            B[k] -= mu * B[j]
        
        Q, R = gram_schmidt(B)  # Update Gram-Schmidt orthogonalization

        # Check Lovász condition
        if np.dot(Q[k-1], Q[k-1]) * delta > np.dot(Q[k], Q[k]) + np.dot(Q[k-1], B[k])**2 / np.dot(Q[k-1], Q[k-1]):
            # Swap vectors
            B[[k, k-1]] = B[[k-1, k]]
            Q, R = gram_schmidt(B)  # Recompute after swapping
            k = max(k-1, 1)  # Go back one step
        else:
            k += 1
    return B

# Function to perform lattice reduction (Placeholder using Gram-Schmidt)
def RED(B):
    Q, R = np.linalg.qr(B.T)
    return Q.T
    return lll_reduction(B)

# Function to orthogonalize and ensure positive diagonals
def ORTH(B):
    Q, R = np.linalg.qr(B)
    return Q @ np.diag(np.sign(np.diag(R)))

# Function to compute the Normalized Second Moment (NSM)
def compute_nsm(B):
    # Volume of the Voronoi region
    V = np.prod(np.diag(B))
    # Compute the NSM using the definition
    n = B.shape[0]
    norms = []
    for _ in range(10000):  # Sample a large number of random points
        z = URAN(n)
        y = z - CLP(B, z @ B)
        e = y @ B
        norms.append(np.linalg.norm(e) ** 2)
    norms = np.array(norms)
    nsm = norms.mean() / (n * (V ** (2 / n)))
    return nsm

# Iterative lattice construction (Algorithm 1)
def iterative_lattice_construction(n, T, Tr, mu0, nu):
    print(f"Initializing lattice construction for dimension {n}...")
    
    # Step 1: Initialize B
    B = ORTH(RED(GRAN(n, n)))
    print("Initial generator matrix B:")
    print(B)

    # Step 2: Compute volume
    V = np.prod(np.diag(B))

    # Step 3: Normalize B
    B = B / (V ** (1 / n))

    print("Starting optimization process...")
    # Main loop
    for t in range(T):
        # Step 5: Update annealing parameter
        mu = mu0 * nu ** (-t / (T - 1))

        # Step 6: Generate random vector
        z = URAN(n)

        # Step 7: Compute y
        y = z - CLP(B, z @ B)

        # Step 8: Compute e
        e = y @ B

        # Step 9-14: Update B
        for i in range(n):
            for j in range(i):
                B[i, j] -= mu * y[i] * e[j]
            B[i, i] -= mu * (y[i] * e[i] - np.linalg.norm(e) ** 2 / (n * B[i, i]))

        # Step 15-19: Periodic reduction and normalization
        if t % Tr == Tr - 1:
            B = ORTH(RED(B))
            V = np.prod(np.diag(B))
            B = B / (V ** (1 / n))
        
        # Log progress every 10% of iterations
        if t % 100 == 0 or t == T - 1:
            nsm = compute_nsm(B)
            print(f"Iteration {t + 1}/{T}, Current NSM: {nsm:.6f}")

    print("Optimization completed!")
    return B

# Training and evaluation
def train_lattice(n, T=1000, Tr=100, mu0=0.01, nu=200):
    B = iterative_lattice_construction(n, T, Tr, mu0, nu)
    print("Final generator matrix B:")
    print(B)

    # Compute the final NSM
    final_nsm = compute_nsm(B)
    print(f"Final NSM for the constructed lattice: {final_nsm:.6f}")
    return B, final_nsm

# Example usage
if __name__ == "__main__":
    n = 3  # Dimension of the lattice
    T = 10000  # Number of iterations
    Tr = 100  # Reduction interval
    mu0 = 0.01  # Initial step size
    nu = 200  # Annealing ratio

    print("Training lattice generator...")
    B, final_nsm = train_lattice(n, T, Tr, mu0, nu)
    print(f"Generated generator matrix B:\n{B}")
    print(f"Achieved Normalized Second Moment (NSM): {final_nsm:.6f}")
