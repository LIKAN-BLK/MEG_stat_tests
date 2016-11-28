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


def vis_heads_array(main_title,sensor_type,*args):
    #This function receives list of data vectors and visualise them as several "heads" (topographies) on one image
    # @main_title - title of whole picture
    # @sensor_type - 'grad' or 'mag'
    # @args - list of tuples, each tuple consists from title and data vector. Example: (head_title,head_data)
    layout = find_layout(info, ch_type= sensor_type)
    number_of_heads = len(args)
    tmp = [list(l) for l in zip(*args)]
    titles = tmp[0]
    data = tmp[1]
    max_row_lenght = 5 #depends from monitor length (:
    fig,axes=plt.subplots(-(-number_of_heads//max_row_lenght),min(max_row_lenght,number_of_heads),figsize=(20, 20))
    fig.suptitle(main_title, fontsize=16)
    min_value = np.array(map(lambda x:x.min(),data)).min()
    max_value = np.array(map(lambda x:x.max(),data)).max()
    max_value = max(abs(min_value),abs(max_value),key=abs)
    for i in range(number_of_heads):
        axes[np.unravel_index(i,axes.shape)].set_title(titles[i])
        if data[i].any():
            im,_ = plot_topomap(data[i],layout.pos,axes=axes[np.unravel_index(i,axes.shape)],
                                vmin=-max_value,vmax=max_value,image_interp='none',show=False)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.3,fraction=0.025)
    return fig


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



