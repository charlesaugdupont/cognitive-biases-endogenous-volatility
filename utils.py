from scipy.ndimage import gaussian_filter
from scipy.stats.qmc import LatinHypercube
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import basinhopping
from collections import Counter
from model import *

def generate_samples(num_samples, num_params, seed):
    lh = LatinHypercube(d=num_params, seed=seed)
    sample = lh.random(n=num_samples)
    return sample

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

def get_minima(interpolator, count_threshold=2, num_points=25, step_size=50, N=200):

    def func(xy):
        x, y = xy
        return interpolator(x, y)[0][0]

    results = []
    for i in np.linspace(0, N-1, num_points):
        for j in np.linspace(0, N-1, num_points):
            init = [i,j]
            minimizer_kwargs = { "method": "L-BFGS-B", "bounds":((0,N),(0,N))}
            R = basinhopping(func, init, minimizer_kwargs=minimizer_kwargs, stepsize=step_size)
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

def get_minima_alternate(interpolator, count_threshold=2, num_points=15, step_size=5, N=200):

    def func(xy):
        x, y = xy
        return interpolator(x,y)[0][0]

    results = []
    coords_list = np.linspace(1, N, num_points)
    for i in coords_list:
        for j in coords_list:
            init = [i, j]
            minimizer_kwargs = {"method": "L-BFGS-B", "bounds": ((1, N), (1, N))}
            R = basinhopping(func, init, minimizer_kwargs=minimizer_kwargs, stepsize=step_size)
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

    # Sort minima by count (descending)
    minima.sort(key=lambda x: x[1], reverse=True)

    # Filter out close minima
    filtered_minima = []
    for item in minima:
        if len(filtered_minima) == 0:
            filtered_minima.append(item)
        else:
            coords = item[0]
            too_close = False
            for elem in filtered_minima:
                if abs(elem[0][0] - coords[0]) < 5 and abs(elem[0][1] - coords[1]) < 5:
                    too_close = True
                    break
            if not too_close:
                filtered_minima.append(item)

    return filtered_minima
