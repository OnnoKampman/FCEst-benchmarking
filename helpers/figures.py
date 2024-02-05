import seaborn as sns


def set_size(width: float = 360., fraction: float = 1, subplots=(1, 1)) -> tuple:
    """
    Set figure dimensions to avoid scaling in LaTeX.

    Source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    width: float or string
            Document width in points (pts), or string of predefined document type
            The default value of 360 is obtained from LaTeX.
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    fig_width_pt = width * fraction  # Width of figure (in pts)
    inches_per_pt = 1 / 72.27  # Convert from pt to inches

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width and height in inches
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    print(f"figsize=({fig_width_in:.2f}, {fig_height_in:.2f})")

    return fig_width_in, fig_height_in


def get_filetype(file_format: str) -> str:
    if file_format == 'raster':
        return '.png'
    else:
        # Vector graphic format.
        return '.pdf'  # or .svg


def get_significance_asterisks(pval: float) -> str:
    if pval <= 0.001:
        return '***'
    elif pval <= 0.01:
        return '**'
    elif pval <= 0.05:
        return '*'
    else:
        return None


def get_ylim(tvfc_summary_measure: str) -> list:
    """
    Set colorbar minimum and maximum.
    """
    match tvfc_summary_measure:
        case 'mean':
            ylim = [-0.05, 1.0]
        case 'variance':
            ylim = [0.0, 0.07]
        case 'std':
            ylim = [0.0, 0.3]
        case 'rate_of_change':
            ylim = [0.0, 0.2]
        case 'ar1':
            ylim = [-1.0, 1.0]
        case _:
            raise NotImplementedError(f"Summary measure '{tvfc_summary_measure:s}' not recognized.")
    return ylim


def get_effect_sizes_xlim(tvfc_summary_measure: str) -> list:
    """
    Set x-axis minimum and maximum.
    """
    match tvfc_summary_measure:
        case 'mean':
            xlim = [-0.38, 0.15]
            asterisk_height = 0.12
        case 'variance':
            xlim = [-0.34, 0.32]
            asterisk_height = 0.29
        case 'std':
            xlim = [-0.34, 0.32]
            asterisk_height = 0.29
        case 'rate_of_change':
            xlim = [-0.14, 0.38]
            asterisk_height = 0.35
        case 'ar1':
            xlim = [-0.14, 0.38]
            asterisk_height = 0.35
        case _:
            raise NotImplementedError(f"Summary measure '{tvfc_summary_measure:s}' not recognized.")
    return xlim, asterisk_height


def get_palette(models_list: list):

    # cmap = matplotlib.colormaps['tab10']

    palette_tab10 = sns.color_palette(palette="deep", n_colors=10)

    if 'DCC-BL' not in models_list:
        # Truncate palette to assure colors are consistent across plots.
        palette_tab10 = sns.color_palette([
            palette_tab10[0], palette_tab10[1],
            palette_tab10[3], palette_tab10[4], palette_tab10[5], palette_tab10[6],
        ])

    return palette_tab10
