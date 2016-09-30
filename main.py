from load_data import get_data
import sys
from os.path import join
from mne.time_frequency import cwt_morlet
import numpy as np
from scipy.stats import ttest_ind

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from scipy.io import savemat

def tft_transofrm(source):
    # average : bool - averaging of output data in sliding windows
    # average_w_width sliding window width
    # average_w_step sliding window step

    window_start = 820 #100ms after fuxation
    window_end = window_start+400


    freqs = range(10,13,1)
    sfreq = 1000
    res = np.zeros((source.shape[0],source.shape[1],len(freqs),source.shape[2]),dtype=np.float32)
    for i in xrange(source.shape[0]):
         res[i,:,:,:] = np.absolute(cwt_morlet(source[i,:,:], sfreq, freqs, use_fft=True, n_cycles=7.0, zero_mean=True, decim=1)).astype('float32',casting='same_kind')
    return res

def baseline_correction(data):
    baseline_start = 100
    baseline_end = baseline_start+300
    res = np.zeros(data.shape,dtype=np.float32)
    #TODO VECTORISE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for i in xrange(data.shape[0]):
        tf_magnitude_baseline = np.log10(data[i,:,:,baseline_start:baseline_end].mean(axis=2))
        res[i,:,:,:] = np.log10(data[i,:,:,:]) - tf_magnitude_baseline[:,:,None]
    return res

def calc_t_stat(target_data, nontarget_data):
    res = ttest_ind(target_data,nontarget_data,axis=0,equal_var=False)
    return res

def visualise(data,title):
    #Plot 2D data as a image
    fig = plt.figure()
    plt.title(title)
    plt.imshow(data,aspect='auto')
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Channel')
    return fig

def vis_each_freq(data,title):
    # visualise data for each frequency and save it as a file
    # data in format channel x freq x time

    if not os.path.isdir('results'):
        os.mkdir('results')
    if not os.path.isdir(os.path.join('results',title)):
        os.mkdir(os.path.join('results',title))
    else:
        file_names = [os.path.join('results',title,file_name) for file_name in os.listdir(os.path.join('results',title))]
        map(os.remove,file_names)

    for fq in range(data.shape[1]):
        fig=visualise(data[:,fq,:],'%s fq=%f' %(title,fq))
        plt.savefig(os.path.join('results',title,'_fq=%0.1f.png' % fq))
        plt.close(fig)

def save_mat(data,title):
    if not os.path.isdir('results'):
        os.mkdir('results')
    savemat(file_name = os.path.join('results',title),mdict=dict(title=data))

def calc_metricts(data_path,sensor_type):
    #Loading data
    #data start time = -820
    target_data, nontarget_data = get_data(data_path,sensor_type) #trials x channels x times
    sensor_type = sensor_type.split(' ')[-1]

    first_target = tft_transofrm(target_data) # trials x channels x freqs x times
    first_nontarget = tft_transofrm(nontarget_data)

    second_target = baseline_correction(first_target)
    second_nontarget = baseline_correction(first_nontarget)

    # Calc mean for UNCORRECTED data
    third_target = first_target.mean(axis=0)
    third_nontarget = first_nontarget.mean(axis=0)
    save_mat(third_target,'third_target_%s' %sensor_type)
    save_mat(third_nontarget,'third_nontarget_%s' %sensor_type)
    vis_each_freq(third_target,'mean_target_notcorrected_%s' %sensor_type)
    vis_each_freq(third_nontarget,'mean_nontarget_notcorrected_%s' %sensor_type)

    # Calc mean for CORRECTED data
    fourth_target = second_target.mean(axis=0)
    fourth_nontarget = second_nontarget.mean(axis=0)
    save_mat(fourth_target,'fourth_target_%s' %sensor_type)
    save_mat(fourth_nontarget,'fourth_nontarget_%s' %sensor_type)
    vis_each_freq(fourth_target,'mean_target_corrected_%s' %sensor_type)
    vis_each_freq(fourth_nontarget,'mean_nontarget_corrected_%s' %sensor_type)

    # Calc t-stat for UNCORRECTED data
    fivth = ttest_ind(first_target,first_nontarget,axis=0,equal_var=False)
    vis_each_freq(fivth.statistic,'t-stat_notcorrected_%s' %sensor_type)
    save_mat(fivth.statistic,'fivth_%s' %sensor_type)

    # Calc t-stat for CORRECTED data
    sixth = ttest_ind(second_target,second_nontarget,axis=0,equal_var=False)
    vis_each_freq(sixth.statistic,'t-stat_corrected_%s' %sensor_type)
    save_mat(sixth.statistic,'sixth_%s' %sensor_type)


if __name__=='__main__':
    exp_num=sys.argv[1]
    path = join('..', 'meg_data1',exp_num)

    calc_metricts(path,'MEG GRAD')

    calc_metricts(path,'MEG MAG')
