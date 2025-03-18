from scipy.ndimage import gaussian_filter
from scipy.stats.qmc import LatinHypercube
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import basinhopping
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from model import *

def generate_samples(num_samples):
    lh = LatinHypercube(d=5)
    sample = lh.random(n=num_samples)
    sample[:, 1] = np.ceil(sample[:,1]*10).astype(int)
    sample[:, 2] = np.ceil(sample[:,2]*10).astype(int)
    sample[:, 3] = np.ceil(sample[:,3]*4).astype(int)/4
    return sample

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

def plot_policy_boundary(policy, params, colors=None, savepath=None):
    plt.figure(figsize=(7, 5))  # Increase figure size

    # Create a custom colormap with the specified colors
    if colors is None:
        colors = ["blue", "red"]
    cmap = ListedColormap(colors)

    # Use imshow with origin='lower' to invert the y-axis
    plt.imshow(policy, cmap=cmap, interpolation='nearest', origin='lower')

    # Create a color bar
    cbar = plt.colorbar(ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(["Save", "Invest"])

    # Set the tick positions and labels
    plt.xticks(np.arange(0, params["N"]-1, 20), np.arange(0, params["N"]-1, 20))
    plt.yticks(np.arange(0, params["N"]-1, 20), np.arange(0, params["N"]-1, 20))

    # Label the axes
    plt.ylabel("Wealth")
    plt.xlabel("Health")

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')

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

def plot_wealth_health_distribution(wealth, health, N):
    heatmap, xedges, yedges = np.histogram2d(wealth, health, bins=75, range=[[0, N], [0, N]])
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap.T, origin='lower', cmap='viridis', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], norm="log")
    plt.colorbar(label='Count')
    plt.xlabel('Wealth')
    plt.ylabel('Health')
    plt.ylim(0, N)
    plt.xlim(0, N)
    plt.show()

def get_interpolation_function(wealth, health, bins=50):
    hist, xedges, _ = np.histogram2d(wealth, health, bins=bins, range=[(0,200), (0,200)])
    density = hist / np.sum(hist)
    with np.errstate(divide='ignore'):
        potential = -np.log(density)
        potential[np.isinf(potential)] = np.nan

    # interpolation function
    max_finite_value = np.nanmax(potential)
    potential[np.isnan(potential)] = max_finite_value + 1
    smoothed_potential = gaussian_filter(potential, sigma=1)
    interp_func = RectBivariateSpline(np.linspace(0,199,len(xedges)-1), np.linspace(0,199,len(xedges)-1), smoothed_potential)
    return interp_func

def get_minima(interpolator, count_threshold=2, num_points=25):

    def func(xy):
        x, y = xy
        return interpolator(x, y)[0][0]

    results = []
    for i in np.linspace(0, 199, num_points):
        for j in np.linspace(0, 199, num_points):
            init = [i,j]
            minimizer_kwargs = { "method": "L-BFGS-B", "bounds":((0,200),(0,200))}
            R = basinhopping(func, init, minimizer_kwargs=minimizer_kwargs, stepsize=50)
            results.append(tuple(R.x.round(2)))

    most_common = Counter(results).most_common()
    minima = []
    for item in most_common:
        count = item[1]
        if len(minima):
            if count >= count_threshold:
                minima.append(item)
            else:
                break
        else:
            minima.append(item)

    return minima
