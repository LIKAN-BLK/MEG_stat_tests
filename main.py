from load_data import get_data
import sys
from mne.time_frequency import cwt_morlet
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import os
from scipy.io import savemat,loadmat

from vis_heads import vis_heads_array



def tft_transofrm(source,freqs):
    # average : bool - averaging of output data in sliding windows
    # average_w_width sliding window width
    # average_w_step sliding window step

    sfreq = 1000
    res = np.zeros((source.shape[0],source.shape[1],len(freqs),source.shape[2]),dtype=np.float32)
    for i in xrange(source.shape[0]):
         res[i,:,:,:] = np.absolute(cwt_morlet(source[i,:,:], sfreq, freqs, use_fft=True, n_cycles=7.0, zero_mean=True, decim=1)).astype('float32',casting='same_kind')
    return res

def baseline_correction(data,epoch_start_time):
    baseline_start = -720
    baseline_end = -720+300
    start_base_index = max((baseline_start - epoch_start_time),0)
    end_base_index = baseline_end - epoch_start_time
    # res = np.zeros(data.shape,dtype=np.float32)
    for i in xrange(data.shape[0]):
        tf_magnitude_baseline = np.log10(data[i,:,:,start_base_index:end_base_index].mean(axis=2))
        data[i,:,:,:] = np.log10(data[i,:,:,:]) - tf_magnitude_baseline[:,:,None]
    return data

def calc_t_stat(target_data, nontarget_data):
    res = ttest_ind(target_data,nontarget_data,axis=0,equal_var=False)
    return res

def save_large_data(data,path_to_dir,freqs):
    # @data have to be  [trials x channels x freqs x times]
    os.mkdir(path_to_dir)
    for fq in range(len(freqs)):
        savemat(file_name = os.path.join(path_to_dir,'%d' %freqs[fq]),mdict=dict(data=data[:,:,fq,:]))

def read_large_data(path_to_dir):
    files = os.listdir(path_to_dir)
    file = files[0]
    files = files[1:]
    data_sample = loadmat(os.path.join(path_to_dir,file))
    res = np.empty((data_sample.shape[0],data_sample.shape[1],0,data_sample.shape[3]),np.float32)
    for file in files:
        res = np.concatenate(res,loadmat(os.path.join(path_to_dir,file))[:,:,np.newaxis,:],axis=2)
    return res


def vis_space_freq(data,title,freqs):
    #Plot 2D data as a image
    fig = plt.figure()
    plt.title(title)
    plt.imshow(data,aspect='auto')
    plt.colorbar()
    plt.xlabel('Frequency')
    plt.ylabel('Channel')
    axes = plt.gca()
    # axes.set_xticklabels(freqs)
    axes.set_xticks(range(len(freqs)))
    axes.set_xticklabels(freqs)
    axes.set_ylim([0,data.shape[0]-1])
    return fig


def save_results(data,freqs,sensor_type,title,result_path):
    mat_path = os.path.join(result_path,'mat')
    rect_img_path = os.path.join(result_path, 'rectangle_images')
    heads_img_path = os.path.join(result_path, 'heads_images')
    if not os.path.isdir(mat_path): os.makedirs(mat_path)
    if not os.path.isdir(rect_img_path): os.makedirs(rect_img_path)
    if not os.path.isdir(heads_img_path): os.makedirs(heads_img_path)
    savemat(file_name = os.path.join(mat_path,title),mdict=dict(data=data)) #save data as .mat file with 'data' variable inside [channelas x freqs x times]
    fig = vis_space_freq(data, title, freqs)
    plt.savefig(os.path.join(rect_img_path, title  + '.png'))
    plt.close(fig)
    heads = [('%.02f Hz' % freqs[fq], data[:, fq]) for fq in range(data.shape[1])]
    fig = vis_heads_array('%s ms' % title, sensor_type.lower(), *heads)
    fig.savefig(os.path.join(heads_img_path, '%s_heads.png' % title))
    plt.close(fig)



def get_tft_data(data,data_type,data_path,sensor_type,freqs,save_tft,load_existing_tft):
    #This function tries to load tft data if flag @load_existing_tft true and data exists,
    #if not, it calculates them from @data arg and save them if @save_tft true
    #@data_type can  be 'target' or 'nontarget'
    expected_dirname_path = os.path.join(data_path,'%s_BTS_%d_%d_%d_%s' %(sensor_type,freqs[0],freqs[-1],freqs[1] - freqs[0],data_type))
    if load_existing_tft & os.path.isdir(expected_dirname_path):
        res=read_large_data(expected_dirname_path)
    else:
        res = tft_transofrm(data,freqs) # trials x channels x freqs x times
        if save_tft:
            save_large_data(res,expected_dirname_path,freqs) # BTS is a hint for next 3 digits - Bottom,Top,Step
    return res # trials x channels x freqs x times



def calc_metrics(data_path,result_path,sensor_type,freqs):
    erase_dir(result_path)
    data = get_data(data_path,sensor_type) #trials x channels x times
    sensor_type = sensor_type.split(' ')[-1]

    data = get_tft_data(data,'target',data_path,sensor_type,freqs,save_tft=False,load_existing_tft=False) # trials x channels x freqs x times

    win_length = 300 #ms
    left_border = -4300 #ms
    right_border = 200 #ms
    right_border_ind = right_border - left_border

    t_val = []
    p_val = []

    # np.seterr(all='raise')
    for ind,t in enumerate(np.arange(0,right_border_ind-win_length+1,win_length)):

        loop_t_val,loop_p_val = ttest_ind(data[:,:,:,t:t+win_length].mean(axis=3),
                                          data[:,:,:,right_border_ind:right_border_ind+win_length].mean(axis=3) ,
                                          axis=0,equal_var=True)
        t_val.append(loop_t_val)
        p_val.append(loop_p_val)
        title = '%d_%d' %(t+left_border,t+left_border+win_length)
        save_results(loop_t_val,freqs,sensor_type, title, result_path)



def erase_dir(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

if __name__=='__main__':
    exp_num=sys.argv[1]
    data_path = os.path.join('..', 'MEG_LONG1', exp_num)

    debug = (sys.argv[2] == 'debug')
    if debug:
        freqs = range(10,15,1)
        res_dir = 'tmp'
        erase_dir(res_dir)
    else:
        freqs = range(10,100,5)
        res_dir = 'results'

    result_path = os.path.join(res_dir,exp_num,'GRAD','fix_vs_nonfix')
    calc_metrics(data_path,result_path,'MEG GRAD',freqs)

    result_path = os.path.join(res_dir,exp_num,'MAG','fix_vs_nonfix')
    calc_metrics(data_path,result_path,'MEG MAG',freqs)
