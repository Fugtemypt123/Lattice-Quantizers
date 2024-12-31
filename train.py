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
# Algorithm 5 in https://ieeexplore.ieee.org/document/5773010
def CLP(r: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    Args:
        r (ndarray): The received vector.
        G (ndarray): The generator matrix of the lattice.
    """
    def sign(double):
        return 1 if double > 0 else -1
    n = G.shape[0]
    C = np.inf  # the radius of the sphere
    i = n
    d = [n - 1 for _ in range(n)]  # d[i] means F[d[i]:n, i] are computed.
    F = np.zeros((n, n), dtype=np.float64); F[n - 1] = r  # F[j, i] = r[i] - u[j+1]G[j+1,i] - ... - u[n-1]G[n-1,i]
    lam = [0 for _ in range(n + 1)]  # lambda[i] = y[i] ** 2 + ... + y[n-1] ** 2
    u = np.zeros(n, dtype=np.int32)
    p = np.zeros(n, dtype=np.float64)  # p[i] = F[i, i] / G[i, i]
    delta = np.zeros(n, dtype=np.float64)
    hat_u = None
    while True:
        while True:  # greedily take the nearest integer
            if i > 0:
                i -= 1
                for j in range(d[i], i, -1):
                    F[j - 1, i] = F[j, i] - u[j] * G[j, i]
                p[i] = F[i, i] / G[i, i]
                u[i] = int(p[i] + 0.5)  # search from the nearest integer
                y = (p[i] - u[i]) * G[i, i]
                delta[i] = sign(y)  # delta[i] determines the direction of the search
                lam[i] = lam[i + 1] + y * y
            else:
                hat_u = u.copy()
                C = lam[0]
            if lam[i] >= C:  # search constraint
                break
        m = i
        while True:  # backtracking until the search constraint is satisfied
            if i < n - 1:
                i += 1
                u[i] += delta[i]
                delta[i] = -delta[i] - sign(delta[i])  # it can be +1, -2, +3, ..., searching from the nearest to the farthest
                y = (p[i] - u[i]) * G[i, i]
                lam[i] = lam[i + 1] + y * y
            else:
                return hat_u
            if lam[i] < C:
                break
        for j in range(m, i):
            if d[j] < i:
                d[j] = i
            else:
                break
    return None  # unreachable

# Function to perform LLL reduction
def gram_schmidt(B):
    """Perform Gram-Schmidt orthogonalization."""
    n = B.shape[0]
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
    """Perform LLL (Lenstra–Lenstra–Lovász) reduction."""
    B = B.copy()
    n = B.shape[0]
    Q, R = gram_schmidt(B)  # Perform initial Gram-Schmidt orthogonalization

    k = 1
    while k < n:
        # Size reduction step
        for j in range(k-1, -1, -1):
            mu = np.dot(B[k], Q[j]) / np.dot(Q[j], Q[j])
            B[k] -= np.round(mu) * B[j]

        # Check Lovász condition
        if np.dot(Q[k-1], Q[k-1]) * delta > (np.dot(Q[k], Q[k]) + (np.dot(Q[k-1], B[k]) ** 2) / np.dot(Q[k-1], Q[k-1])):
            # Swap vectors
            B[[k, k-1]] = B[[k-1, k]]
            Q, R = gram_schmidt(B)  # Recompute after swapping
            k = max(k-1, 1)
        else:
            k += 1
    return B

def RED(B):
    """Wrapper for reduction, currently uses LLL reduction."""
    return lll_reduction(B)

# Function to orthogonalize and ensure positive diagonals
def ORTH(B):
    """Ensure lower triangular matrix with positive diagonal elements."""
    # try:
    #     L = np.linalg.cholesky(B)
    #     return L
    # except np.linalg.LinAlgError:
    B = np.tril(B)  # Enforce lower triangular form
    diag_sign = np.sign(np.diag(B))
    diag_sign[diag_sign == 0] = 1  # Replace zero diagonals with positive
    B = B * diag_sign[:, np.newaxis]  # Adjust rows to ensure diagonals are positive
    return B

# Function to compute the Normalized Second Moment (NSM)
def compute_nsm(B):
    # Volume of the Voronoi region
    V = np.prod(np.diag(B))
    # Compute the NSM using the definition
    n = B.shape[0]
    norms = []
    for _ in range(10000):  # Sample a large number of random points
        z = URAN(n)
        y = z - CLP(z @ B, B)
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
        y = z - CLP(z @ B, B)

        # Step 8: Compute e
        e = y @ B

        # Step 9-14: Update B
        for i in range(n):
            for j in range(i):
                B[i, j] -= mu * y[i] * e[j]
            B[i, i] -= mu * (y[i] * e[i] - np.linalg.norm(e) ** 2 / (n * B[i, i]))

        # Step 15-19: Periodic reduction and normalization
        # if t % Tr == Tr - 1:
        #     B = ORTH(RED(B))
        #     V = np.prod(np.diag(B))
        #     B = B / (V ** (1 / n))
        
        # Log progress every 10% of iterations
        if t % 10 == 0 or t == T - 1:
            nsm = compute_nsm(B)
            print(f"Iteration {t + 1}/{T}, Current NSM: {nsm:.6f}")
        if t % 100 == 0 or t == T - 1:
            print(f"Iteration {t + 1}/{T}, Current B: {B}")

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
    n = 10  # Dimension of the lattice
    T = 100000  # Number of iterations
    Tr = 100  # Reduction interval
    mu0 = 0.01  # Initial step size
    nu = 200  # Annealing ratio

    print("Training lattice generator...")
    B, final_nsm = train_lattice(n, T, Tr, mu0, nu)
    print(f"Generated generator matrix B:\n{B}")
    print(f"Achieved Normalized Second Moment (NSM): {final_nsm:.6f}")
