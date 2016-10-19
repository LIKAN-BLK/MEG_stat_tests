from mne.viz import plot_topomap
import numpy as np
from mne.channels import find_layout
from mne.io import read_raw_fif
from mne.datasets import sample
import os
import matplotlib.pyplot as plt


info = read_raw_fif(sample.data_path() + '/MEG/sample/sample_audvis_raw.fif',verbose=False).info

def save_fig(path,exp_num,title,fig):
    if not os.path.isdir(os.path.join('results',exp_num)):
        os.mkdir(os.path.join('results',exp_num))
    fig.savefig(os.path.join('results',exp_num,title + '.png'))

def visualise(t_data,p_data,sensor_type,freq):
    #reseives t-values (and optionaly p-values) as vectors [channel x 1]
    layout = find_layout(info, ch_type=sensor_type)
    im,_ = plot_topomap(data= t_data,pos = layout.pos,show=False)
    plt.colorbar(im)
    title = 'fq=%d_min_p=%0.2f' %(freq,p_data.min())
    plt.title(title)
    return title
    # save_fig(exp_num,main_title,fig)

def save_heads(heads_path,t_data,p_data,sensor_type,freqs):
     if not os.path.isdir(heads_path):
        os.makedirs(heads_path)

     for freq_indx in range(t_data.shape[1]):
        title = visualise(t_data[:,freq_indx],p_data[:,freq_indx],sensor_type,freqs[freq_indx])
        plt.savefig(os.path.join(heads_path,title + '.png'))
        plt.close()

