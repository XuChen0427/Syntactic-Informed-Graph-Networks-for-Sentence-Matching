import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

def matrix_visualization(text1,text2,matrix,store_filename):

    df = pd.DataFrame (matrix, columns=text2, index=text1)

    fig = plt.figure ()

    ax = fig.add_subplot (111)

    cax = ax.matshow (df, interpolation='nearest', cmap='hot_r')
    # cax = ax.matshow(df)
    fig.colorbar (cax)

    tick_spacing = 1
    ax.xaxis.set_major_locator (ticker.MultipleLocator (tick_spacing))
    ax.yaxis.set_major_locator (ticker.MultipleLocator (tick_spacing))

    # fontdict = {'rotation': 'vertical'}    #设置文字旋转
    fontdict = {'rotation': 90}  # 或者这样设置文字旋转
    # ax.set_xticklabels([''] + list(df.columns), rotation=90)  #或者直接设置到这里
    # Axes.set_xticklabels(labels, fontdict=None, minor=False, **kwargs)
    ax.set_xticklabels ([''] + list (df.columns), fontdict=fontdict)
    ax.set_yticklabels ([''] + list (df.index))

    plt.show ()
    plt.savefig (store_filename)
