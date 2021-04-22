import numpy as np
from matplotlib import pyplot as plt

from sklearn import mixture
import scipy.stats as stats

_W = 10
_H = 5

def plotPeaks(peaks, ts, title, xlabel='Time', ylabel='Count'):
    fig = plt.figure(figsize=(_W, _H))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ts, peaks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def plotGaussianFilter(g_in, g_out, title="", xlabel='bpm', ylabel='Count'):
    fig = plt.figure(figsize=(_W, _H))
    ax = fig.add_subplot(1, 1, 1)
    #plt.plot(g_in)
    ax.hist(g_in, bins=240, histtype='bar', density=False, alpha=0.5)
    plt.plot(g_out)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    #plt.show()

def plotHist(occ_g, title="", xlabel="bpm", ylabel='Count'):
    fig = plt.figure(figsize=(_W, _H))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(occ_g.ravel(), bins=200, histtype='bar', density=False, alpha=0.5)

    ax.set(xlim=(0, 240))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def plotGaussianAndHist(occ_g, ws, ms, cs, n_comp, title="", xlabel='Time', ylabel='Count'):
    f_axis = occ_g.copy().ravel()
    f_axis.sort()
    fig = plt.figure(figsize=(_W, _H))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(occ_g.ravel(), bins=200, histtype='bar', density=True, alpha=0.5)
    for j in range(n_comp):
        ax.plot(f_axis,ws[j]*stats.norm.pdf(f_axis,ms[j],np.sqrt(cs[j])).ravel(), c='red')

    ax.set(xlim=(0, 240))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def error(in_list, gt_list):
    for s, gt in zip(in_list, gt_list):
        e = min(np.abs((s-gt))/gt, np.abs((s-2*gt))/gt, np.abs((s-gt/2))/gt, np.abs((s-3*gt))/gt, np.abs((s-gt/3))/gt)
        print (s, gt, e*100)

if __name__ == "__main__":
    error([163, 110, 134, 152], [83.6, 113.5, 68.5, 159.4])
    print ("paranoid android, addicted to love, chariots of filre, mi tonada montuna")
