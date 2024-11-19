import matplotlib.pyplot as plt
import seaborn as sns
from model import *

def plot_utility_transition(util, num_steps):
    fig, axs = plt.subplots(1, 5, figsize=(15,3))
    for k, step in enumerate([10, 50, 100, 500, num_steps]):
        axs[k].scatter(util[:,0], util[:,step-1], c="dodgerblue", s=5, alpha=0.5)
        axs[k].set_xlabel("Utility (t=0)")
        axs[k].set_ylabel(f"Utility (t={step})")
        axs[k].plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), color="black", linestyle="dashed", alpha=0.75)
    fig.tight_layout()
    plt.show()

def plot_utility_trajectories(util):
    plt.figure(figsize=(12, 5))
    plt.plot(util.T, linewidth=0.5, alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('Utility')
    plt.show()

def plot_policy_boundary(policy, params):
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(policy, cmap="rocket")
    ax.invert_yaxis()
    plt.xticks(np.arange(0, params["N"]-1, 20), np.arange(0, params["N"]-1, 20))
    plt.yticks(np.arange(0, params["N"]-1, 20), np.arange(0, params["N"]-1, 20))
    plt.show()


def plot_prob_health_decrease(params):
    arr = np.zeros((params["N"],params["N"]))
    for w in range(params["N"]):
        for h in range(params["N"]):
            arr[w][h] = prob_health_decrease(w, h, params["N"], params["health_decrease_scale"])

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(arr)
    ax.invert_yaxis()
    plt.title("Probability of Health Decrease If Not Investing")
    plt.xticks(np.arange(0, params["N"]-1, 20), np.arange(0, params["N"]-1, 20))
    plt.yticks(np.arange(0, params["N"]-1, 20), np.arange(0, params["N"]-1, 20))
    plt.show()

def plot_agent_w_h_trajectory(policy, params):
    w, h = np.random.randint(0, params["N"], size=2)
    path = [(w, h)]
    for _ in range(100):
        action = policy[w-1][h-1]
        if action == 1:
            h = min(h+params["health_delta"], params["N"]) if np.random.uniform() < params["P_H_increase"] else h
            w = max(w-params["invest_cost"], 1)
        else:
            h = max(h-params["health_delta"], 1) if np.random.uniform() < prob_health_decrease(w, h, params["N"], params["health_decrease_scale"]) else h
            w = min(w+1, params["N"])
        path.append((w, h))

    order = np.arange(len(path))
    x_vals, y_vals = zip(*path)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_vals, y_vals, c=order, cmap='viridis', s=50)
    plt.ylim(0, params["N"])
    plt.xlim(0, params["N"])
    plt.colorbar(scatter, label="Time Step")
    plt.title("Coordinates with Color Gradient from Earliest to Latest Position")
    plt.xlabel("Wealth")
    plt.ylabel("Health")
    plt.show()