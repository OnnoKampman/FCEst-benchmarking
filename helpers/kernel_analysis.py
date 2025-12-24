import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# import statsmodels.api as sm


def plot_lengthscale_window_length_relation(
    config_dict: dict,
    kernel_params_array: np.array,
    optimal_window_lengths_array: np.array,
    figures_savedir: str = None,
) -> None:
    """
    Plots a simple figure showing the relationship between WP kernel lengthscale and SW-CV window length.

    TODO: how do we deal with outliers?

    Sources:
        https://seaborn.pydata.org/generated/seaborn.jointplot.html
        https://seaborn.pydata.org/examples/joint_kde.html
    """
    sns.set(style="whitegrid")
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))

    # Convert axes to seconds.
    n_time_steps = config_dict['n-time-steps']  # N

    kernel_params_array *= config_dict['repetition-time']
    kernel_params_array *= n_time_steps

    optimal_window_lengths_array *= config_dict['repetition-time']
    print(kernel_params_array.shape, optimal_window_lengths_array.shape)

    kernel_params_array = np.squeeze(kernel_params_array)
    optimal_window_lengths_array = np.squeeze(optimal_window_lengths_array)

    # Compute fit.
    r, p = scipy.stats.pearsonr(
        kernel_params_array,
        optimal_window_lengths_array,
    )
    print(f'r = {r:.2f}, p = {p:.2f}')

    g = sns.jointplot(
        x=kernel_params_array,
        y=optimal_window_lengths_array,
        # data=(kernel_params_array, optimal_window_lengths_array),
        kind='reg',  # or 'scatter', 'kde', 'hist', 'hex', 'reg', 'resid'
        height=3.2,  # plot will be square
        # xlim=,
        # ylim=,
        marker="+",
        scatter_kws={
            "s": 8
        },
        marginal_kws=dict(
            bins=40,
            fill=False,
        ),
    )

    g.set_axis_labels('WP kernel lengthscale [s]', 'SW-CV window length [s]')

    # g.figure.tight_layout()

    if figures_savedir is not None:
        plt.savefig(
            os.path.join(figures_savedir, 'lengthscale_optimal_window_length_relations.pdf'),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved lengthscale optimal window length relation figure to '{figures_savedir:s}'.")
        plt.close()
