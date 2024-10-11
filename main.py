import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def transport_cost(Xk, Ck):
    return np.sum(Ck * Xk)

def marginal_constraints(X_list, p, q):
    sum_X = np.sum(X_list, axis=0)
    return np.allclose(np.sum(sum_X, axis=1), p) and np.allclose(np.sum(sum_X, axis=0), q)

def decentralized_optimization(Ck_list, p, q, tol=1e-3, max_iter=100):
    N, n = len(Ck_list), Ck_list[0].shape[0]
    X_list = [np.random.rand(n, n) for _ in range(N)]
    for iteration in range(max_iter):
        for k in range(N):
            Ck, Xk = Ck_list[k], X_list[k]
            def objective(Xk_flat):
                return transport_cost(Xk_flat.reshape(n, n), Ck)
            constraints = [
                {'type': 'eq', 'fun': lambda Xk_flat: np.sum(Xk_flat.reshape(n, n), axis=0) - q},
                {'type': 'eq', 'fun': lambda Xk_flat: np.sum(Xk_flat.reshape(n, n), axis=1) - p},
            ]
            res = minimize(objective, Xk.flatten(), constraints=constraints, tol=tol)
            X_list[k] = res.x.reshape(n, n)
        cost_diffs = np.array([transport_cost(X_list[k], Ck_list[k]) for k in range(N)])
        if np.allclose(cost_diffs, cost_diffs[0], atol=tol) and marginal_constraints(X_list, p, q):
            print(f"Converged after {iteration + 1} iterations")
            break
    return X_list

def plot_transportation_map(X_list):
    N = len(X_list)
    fig, axs = plt.subplots(1, N, figsize=(4 * N, 4))
    for k in range(N):
        ax = axs[k]
        Xk = X_list[k]
        cax = ax.imshow(Xk, cmap='viridis', interpolation='none')
        fig.colorbar(cax, ax=ax)
        ax.set_title(f'Agent {k + 1} Transportation Plan')
        ax.set_xlabel('Destination')
        ax.set_ylabel('Source')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    N, n = 3, 5
    Ck_list = [np.random.rand(n, n) for _ in range(N)]
    p, q = np.random.rand(n), np.random.rand(n)
    p, q = p / np.sum(p), q / np.sum(q)
    X_list = decentralized_optimization(Ck_list, p, q)
    
    # Show cost matrices and transportation plans
    for k in range(N):
        print(f"Agent {k + 1} cost matrix C_k:\n{Ck_list[k]}")
        print(f"Agent {k + 1} transportation plan X_k:\n{X_list[k]}")
    
    plot_transportation_map(X_list)
