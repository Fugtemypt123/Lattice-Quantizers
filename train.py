import numpy as np
import argparse
import os

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

def gram_schmidt(basis):
    """
    Perform Gram-Schmidt orthogonalization.
    
    Parameters:
        basis: An (n, m) basis matrix, where n is the dimension and m is the number of basis vectors.
    
    Returns:
        B_star: A matrix of orthogonal basis vectors.
        mu: The computed mu matrix.
    """
    n, m = basis.shape
    B_star = np.zeros((n, m))
    mu = np.zeros((m, m))
    
    for i in range(m):
        B_star[:, i] = basis[:, i]
        for j in range(i):
            mu[i, j] = np.dot(basis[:, i], B_star[:, j]) / np.dot(B_star[:, j], B_star[:, j])
            B_star[:, i] -= mu[i, j] * B_star[:, j]
    
    return B_star, mu

def lll_reduction(basis, delta=0.75):
    """
    Perform the LLL reduction algorithm.
    
    Parameters:
        basis: An (n, m) basis matrix, where n is the dimension and m is the number of basis vectors.
        delta: The δ parameter in the Lovász condition, typically set to 0.75.
    
    Returns:
        basis: The reduced basis matrix.
    """
    n, m = basis.shape
    B_star, mu = gram_schmidt(basis)
    k = 1
    
    while k < m:
        # Size reduction
        for j in range(k-1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                r = round(mu[k, j])
                basis[:, k] -= r * basis[:, j]
                # Update mu
                B_star, mu = gram_schmidt(basis)
        
        # Check the Lovász condition
        B_star_k = np.dot(basis[:, k], B_star[:, k])
        B_star_k1 = np.dot(basis[:, k-1], B_star[:, k-1])
        lhs = delta * (np.dot(basis[:, k-1], basis[:, k-1]))
        rhs = np.dot(basis[:, k], basis[:, k]) + mu[k, k-1]**2 * np.dot(basis[:, k-1], basis[:, k-1])
        
        if lhs <= rhs:
            k += 1
        else:
            # Exchange b_k and b_{k-1}
            basis[:, [k, k-1]] = basis[:, [k-1, k]]
            # Update Gram-Schmidt
            B_star, mu = gram_schmidt(basis)
            k = max(k-1, 1)
    
    return basis

def RED(B):
    return lll_reduction(B)

# Function to orthogonalize and ensure positive diagonals
def ORTH(B):
    """
    Use Cholesky decomposition to rotate and reflect the generating matrix into a lower triangular matrix, ensuring positive diagonal elements.

    Parameters:
        B: A generating matrix of size (n, n).

    Returns:
        L: A lower triangular matrix satisfying A = B B^T = L L^T.
    """
    # Compute the Gram matrix A = B B^T
    A = B @ B.T
    
    # Attempt Cholesky decomposition
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        # If A is not positive definite, force it to be lower triangular and adjust the diagonal to be positive
        L = np.tril(A)
        diag_sign = np.sign(np.diag(L))
        diag_sign[diag_sign == 0] = 1  # Replace zero diagonal elements with positive ones
        L = L * diag_sign[:, np.newaxis]  # Adjust rows to ensure the diagonal is positive
    
    return L

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
def iterative_lattice_construction(n, T, Tr, mu0, nu, use_red):
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
        if t % Tr == Tr - 1 and use_red == True:
            B = ORTH(RED(B))
            print(B)
            V = np.prod(np.diag(B))
            B = B / (V ** (1 / n))
        
        # Log progress every 10 of iterations
        if t % 10 == 0 or t == T - 1:
            nsm = compute_nsm(B)
            print(f"Iteration {t + 1}/{T}, Current NSM: {nsm:.6f}")
        if t % 100 == 0 or t == T - 1:
            # save B to file
            if not os.path.exists(f"lattice_B_{n}"):
                os.makedirs(f"lattice_B_{n}")
            np.save(f"lattice_B_{n}/{t}_nsm={nsm:.6f}.npy", B)

    print("Optimization completed!")
    return B

# Training and evaluation
def train_lattice(n, T=1000, Tr=100, mu0=0.01, nu=200, use_red=False):
    B = iterative_lattice_construction(n, T, Tr, mu0, nu, use_red)
    print("Final generator matrix B:")
    print(B)

    # Compute the final NSM
    final_nsm = compute_nsm(B)
    print(f"Final NSM for the constructed lattice: {final_nsm:.6f}")
    return B, final_nsm

# Example usage
if __name__ == "__main__":
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Dimension of the lattice")
    parser.add_argument("--T", type=int, default=100000, help="Number of iterations")
    parser.add_argument("--Tr", type=int, default=100, help="Reduction interval")
    parser.add_argument("--mu0", type=float, default=0.01, help="Initial step size")
    parser.add_argument("--nu", type=int, default=200, help="Annealing ratio")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--use_red", action="store_true", help="Use LLL reduction or not")
    args = parser.parse_args()

    print("Training lattice generator...")
    np.random.seed(args.seed)
    B, final_nsm = train_lattice(args.n, args.T, args.Tr, args.mu0, args.nu, args.use_red)
    print(f"Generated generator matrix B:\n{B}")
    print(f"Achieved Normalized Second Moment (NSM): {final_nsm:.6f}")
