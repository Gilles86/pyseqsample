import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import stats
import seaborn as sns


def plot_responses(responses, rts, bins=None, t=None, kde=True, **kwargs):
    
    if bins == None:
        bins = np.linspace(0, 1.05 * rts.max(), len(rts)/ 15)
        
    if not t:
        t = np.linspace(0, 1.05*rts.max(), 100)
    
    for r in np.unique(responses):
        idx = responses == r
        
        if kde:
            kde = sp.stats.gaussian_kde(rts[idx])
            plt.plot(t, kde.pdf(t) * idx.mean(), color=sns.color_palette()[r-1], lw=2)

        hist, bin_edges = np.histogram(rts[idx], bins=bins, density=True)        
        plt.bar(bin_edges[:-1], hist * (idx.mean()), width=bins[1] - bins[0], color=sns.color_palette()[r-1], alpha=0.8 )


def plot_quantiles(responses, rts, q=(0.1, 0.3, 0.5, 0.7, 0.9), *args, **kwargs):
    
    q = np.array(q)

    
    for r in np.unique(responses):
        
                
        xs = np.percentile(rts, q*100)
        
        ys = q * (responses == r).mean()
        
        plt.plot(xs, ys, c=sns.color_palette()[r-1], **kwargs)


    plt.xlim(0, 1.05 * rts.max())
    plt.ylim(0, 1)
