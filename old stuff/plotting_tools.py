import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

""" draw ellipse associated to a covariance matrix """
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 5):
        ax.add_patch(patches.Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

""" plot data and corresponding GMM prediction """
def plot_gmm(means, covariances, weights, X, scale = None, labels=None, ax=None):
    ax = ax or plt.gca()

    w_factor = 0.3 / weights.max()
    for pos, covar, w in zip(means, covariances, weights):
        draw_ellipse(pos, covar, alpha=w * w_factor)
    
    if labels is not None:
        if scale is not None:
            ax.scatter(X[:, 0], X[:, 1], c=labels, s=scale, cmap='viridis', zorder=0)
        else:
            ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=0)    
    else:
        if scale is not None:
            ax.scatter(X[:, 0], X[:, 1], c=labels, s=scale, cmap='viridis', zorder=0)
        else:
            ax.scatter(X[:, 0], X[:, 1], c=labels, s= 40, zorder=0)
    ax.set_xlim(min(min(X[:, 0]), min(means[:,0]) ), max(max(X[:, 0]), max(means[:,0]) ))
    ax.set_ylim(min(min(X[:, 1]), min(means[:,1]) ), max(max(X[:, 1]), max(means[:,1]) ))

def plot_kmeans(centers, X, labels):
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
