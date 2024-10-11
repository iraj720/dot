import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Function to compute the transportation cost for a given Xk and Ck
def transport_cost(Xk, Ck):
    return np.sum(Ck * Xk)

# Function to check the marginal constraints
def marginal_constraints(X_list, p, q):
    sum_X = np.sum(X_list, axis=0)
    row_sum = np.sum(sum_X, axis=1)
    col_sum = np.sum(sum_X, axis=0)
    return np.allclose(row_sum, p) and np.allclose(col_sum, q)

# Define the decentralized optimization for each agent
def decentralized_optimization(Ck_list, p, q, tol=1e-3, max_iter=100):
    N = len(Ck_list)
    n = Ck_list[0].shape[0]
    
    # Initialize Xk matrices randomly
    X_list = [np.random.rand(n, n) for _ in range(N)]
    
    # Iteratively optimize each agent's transportation plan
    for iteration in range(max_iter):
        for k in range(N):
            Ck = Ck_list[k]
            Xk = X_list[k]

            # Define objective function to minimize for agent k
            def objective(Xk_flat):
                Xk = Xk_flat.reshape(n, n)
                return transport_cost(Xk, Ck)

            # Define equality constraints for marginal constraints
            constraints = [
                {'type': 'eq', 'fun': lambda Xk_flat: np.sum(Xk_flat.reshape(n, n), axis=0) - q},
                {'type': 'eq', 'fun': lambda Xk_flat: np.sum(Xk_flat.reshape(n, n), axis=1) - p},
            ]

            # Optimize Xk
            res = minimize(objective, Xk.flatten(), constraints=constraints, tol=tol)
            X_list[k] = res.x.reshape(n, n)

        # Check for convergence (based on cost equality and marginal constraints)
        cost_diffs = np.array([transport_cost(X_list[k], Ck_list[k]) for k in range(N)])
        if np.allclose(cost_diffs, cost_diffs[0], atol=tol) and marginal_constraints(X_list, p, q):
            print(f"Converged after {iteration+1} iterations")
            break

    return X_list

# Function to visualize the transportation map
def plot_transportation_map(X_list):
    N = len(X_list)
    fig, axs = plt.subplots(1, N, figsize=(4 * N, 4))

    for k in range(N):
        ax = axs[k]
        Xk = X_list[k]
        cax = ax.imshow(Xk, cmap='viridis', interpolation='none')
        fig.colorbar(cax, ax=ax)
        ax.set_title(f'Agent {k+1} Transportation Plan')
        ax.set_xlabel('Destination')
        ax.set_ylabel('Source')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    N = 3  # Number of agents
    n = 5  # Size of the cost matrix

    # Generate random cost matrices Ck for each agent
    Ck_list = [np.random.rand(n, n) for _ in range(N)]

    # Define p and q (marginals)
    p = np.random.rand(n)
    q = np.random.rand(n)

    # Normalize p and q to sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Perform decentralized optimization
    X_list = decentralized_optimization(Ck_list, p, q)

    # Output results
    for k, Xk in enumerate(X_list):
        print(f"Agent {k+1} transportation plan:\n{Xk}")

    # Plot the transportation map
    plot_transportation_map(X_list)
