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

def visualise(exp_num,main_title,t_data,p_data=None):
    #reseives t-values (and optionaly p-values) as vectors [channel x 1]
    #max
    layout = find_layout(info, ch_type='grad')
    max_t_value = t_data.max()
    fig = plt.figure()
    im,_ = plot_topomap(t_data,layout.pos,fig.gca,show=False)
    plt.colorbar()
    save_fig(exp_num,main_title,fig)
