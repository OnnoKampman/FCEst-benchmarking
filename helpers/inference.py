import logging
import os
import socket

import matplotlib.pyplot as plt
import numpy as np

if socket.gethostname() == 'hivemind':
    import matplotlib
    matplotlib.use('Agg')


def save_elbo_plot(
        maxiter, log_interval: int, logf,
        savedir: str = None, figure_name: str = None, figsize=(12, 6), figure_dpi: int = 100
) -> None:
    """
    Save ELBO plot to check for convergence after training.

    :param maxiter:
    :param log_interval:
    :param logf:
    :param savedir: directory to save figure. If set to 'show', the figure is printed instead.
    :param figure_name:
    :param figsize:
    :param figure_dpi:
    """
    plt.figure(figsize=figsize)
    plt.plot(np.arange(maxiter)[::log_interval], logf)
    plt.xlabel('iteration')
    _ = plt.ylabel('ELBO')
    plt.grid()
    plt.yscale('symlog')
    plt.tight_layout()

    if savedir is not None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, figure_name), dpi=figure_dpi)
        logging.info(f"Saved figure '{figure_name:s}' in '{savedir:s}'.")
        plt.close()
