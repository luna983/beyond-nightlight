import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


def dummy_to_cat(df):
    """Converts one-hot vectors (dummies) to one categorical variable.
    
    This does not implement input checking. It takes the maximum value along each row,
    and returns the name of the column there. Returns NA if any element in the row is NA.
    
    Args:
        df (pandas.DataFrame): data frame of dummies
    
    Returns:
        pandas.Series: converted categorical variable
    """
    return df.apply(lambda x: x.idxmax(), axis=1)


def plot_RD(df, x, y, bandwidth, formula, ylim, yticks,
            xlabel, ylabel, title, out_dir):
    """Plots a regression discontinuity figure.
    
    Args:
        df (pandas.DataFrame): data frame containing x and y variables
        x, y (str): key of the x and y variables
        bandwidth (float): bandwidth of RD, assuming the discontinuity is at 0 for x
            and the range examined will be [-bandwidth, bandwidth] for x
        formula (str): statsmodel R-style formula for fitting regressions on both sides
        ylim (tuple of float)
        yticks (list of float)
        xlabel, ylabel, title (str)
        out_dir (str)
    """
    df_fitted = pd.DataFrame(
        {x: np.linspace(-bandwidth, bandwidth, 101)})
    model = smf.ols(formula, data=df.loc[df[x].abs() < bandwidth, :],
                    missing='drop')
    res = model.fit()
    df_fitted['fit'] = res.predict(exog=df_fitted)
    plt.scatter(df[x], df[y], color='dimgrey', facecolors='none')
    plt.plot(df_fitted.loc[df_fitted[x] > 0, x],
             df_fitted.loc[df_fitted[x] > 0, 'fit'],
             '-', linewidth=2, color='dimgrey')
    plt.plot(df_fitted.loc[df_fitted[x] < 0, x],
             df_fitted.loc[df_fitted[x] < 0, 'fit'],
             '-', linewidth=2, color='dimgrey')
    plt.vlines(0, *ylim, color='firebrick', linewidth=1.5)
    plt.xlim((-bandwidth, bandwidth))
    plt.ylim(ylim)
    plt.yticks(yticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir)
