from mne.viz import plot_topomap
import numpy as np
from mne.channels import find_layout
from mne.io import read_raw_fif
from mne.datasets import sample
import os
import matplotlib.pyplot as plt
from statsmodels.sandbox.stats.multicomp import fdrcorrection0


info = read_raw_fif(sample.data_path() + '/MEG/sample/sample_audvis_raw.fif',verbose=False).info


def visualise(t_data,p_data,sensor_type,freq):
    #reseives t-values (and optionaly p-values) as vectors [channel x 1]
    layout = find_layout(info, ch_type=sensor_type.lower())
    im,_ = plot_topomap(data= t_data,pos = layout.pos,show=False,vmin=-4.5, vmax=4.5)
    plt.colorbar(im)
    title = 'fq=%d_min_p=%0.4f' %(freq,p_data.min())
    plt.title(title)
    return title
    # save_fig(exp_num,main_title,fig)

def save_heads(heads_path,t_data,p_data,sensor_type,freqs):

     if not os.path.isdir(heads_path):
        os.makedirs(heads_path)
     mask,adjusted_p = fdrcorrection0(p_data.flatten(),0.3)
     mask=mask.reshape(p_data.shape)
     t_data[~mask] = 0.001 # Because topomap can't drow zero heads
     for freq_indx in range(t_data.shape[1]):
        title = visualise(t_data[:,freq_indx],p_data[:,freq_indx],sensor_type,freqs[freq_indx])
        plt.savefig(os.path.join(heads_path,title + '.png'))
        plt.close()

def get_heads_from_mat(path_to_mat,file_name_t,file_name_p,freqs):
    from scipy.io import loadmat


    dataset_name = file_name_t.split('_')[0]
    sensor_type = file_name_t.split('_')[-1]

    t_data_path = os.path.join(path_to_mat,file_name_t)
    t_data = loadmat(t_data_path)['data']
    p_data_path = os.path.join(path_to_mat,file_name_p)
    p_data =  loadmat(p_data_path)['data']
    heads_path = os.path.join(path_to_mat,dataset_name + '_heads')
    save_heads(heads_path,t_data,p_data,sensor_type,freqs)

if __name__=='__main__':
    import sys
    exp_num=sys.argv[1]
    freqs = range(10,100,5) #TODO fix this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    path_to_mat = os.path.join('.', 'results',exp_num,'MAG')

    get_heads_from_mat(path_to_mat,'seventh_t_MAG','seventh_p_MAG',freqs)
    get_heads_from_mat(path_to_mat,'eighth_t_MAG','eighth_p_MAG',freqs)

    path_to_mat = os.path.join('.', 'results',exp_num,'GRAD')

    get_heads_from_mat(path_to_mat,'seventh_t_GRAD','seventh_p_GRAD',freqs)
    get_heads_from_mat(path_to_mat,'eighth_t_GRAD','eighth_p_GRAD',freqs)



