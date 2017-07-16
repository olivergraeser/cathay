import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)       
    
def to_histogram(df, column, minval=0, maxval=1, bincount=20, by=None):
    binsize = (maxval-minval)/bincount
    bins = [round(minval + _ * binsize,2) for _ in range(0, int(bincount))]

    if by is None:
        counter = [0]*int(bincount)
        for ix, rw in df.iterrows():
            #import pdb; pdb.set_trace()
            val = max(min(int(rw[column]/binsize),bincount-1),0)
            counter[val] += 1
        return [_/sum(counter) for _ in counter], bins
    else:
        hist_dict = dict()
        for uv in df[by].unique():
            counter = [0]*int(bincount)
            for ix, rw in df[df[by]==uv].iterrows():
                val = max(min(int(rw[column]/binsize),bincount-1),0)
                counter[val] += 1
            hist_dict[uv] = [_/sum(counter) for _ in counter]
        return hist_dict, bins
    
def plot_hist(data,bins):
    maxval = max(sum(data.values(), []))
    lcnt = len(data)
    xlocs = range(len(bins))
    fig,ax = plt.subplots()
    fig.set_size_inches(10.5, 5.5)
    i=0
    for label in data.keys():
        ax.bar([_ + 0.15*(i-1) for _ in xlocs], data[label], width=0.4, color=tableau20[2*i], alpha=0.4)
        plt.text(len(bins)-1, maxval/2 + maxval/lcnt/2*i, label, fontsize=14, color=tableau20[2*i], alpha=0.4)
        i+=1

    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)  
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()   
    plt.ylabel('frequency')
    plt.xlabel('MaxShare')


    
    ax.set_xticks(xlocs[::2])  # set the x ticks to be at the middle of each bar since the width of each bar is 0.8
    ax.set_xticklabels(bins[::2])  #replace the name of the x ticks with your Groups name
    plt.show()
    
def plot_importance(tdc, model, threshold):
    ziplist = sorted(list(zip(tdc, model.feature_importances_)), key=lambda _: -_[1])
    cnt = len([_ for _ in ziplist if _[1]>threshold])
    data = [_[1] for _ in ziplist][:cnt]
    labels = [_[0] for _ in ziplist][:cnt]
    xlocs = list(range(0,cnt))
    fig, ax = plt.subplots()
    fig.set_size_inches(15.5, 7.5)
    ax.bar(xlocs, data, width=0.4, color=tableau20[1], alpha=0.8)
    ax.set_xticks(xlocs)  # set the x ticks to be at the middle of each bar since the width of each bar is 0.8
    ax.set_xticklabels(labels)  #replace the name of the x ticks with your Groups name
    plt.ylabel('importance')
    plt.xlabel('feature')
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)  
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()   
    plt.show()