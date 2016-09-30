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
    res = np.zeros((source.shape[0],source.shape[1],len(freqs),source.shape[2]))
    for i in xrange(source.shape[0]):
         res[i,:,:,:] = np.absolute(cwt_morlet(source[i,:,:], sfreq, freqs, use_fft=True, n_cycles=7.0, zero_mean=True, decim=1))
    return res

def baseline_correction(data):
    baseline_start = 100
    baseline_end = baseline_start+300
    res = np.zeros(data.shape)
    #TODO VECTORISE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for i in xrange(data.shape[0]):
        tf_magnitude_baseline = np.log10(data[i,:,:,baseline_start:baseline_end].mean(axis=2))
        res[i,:,:,:] = np.log10(data[i,:,:,:]) - tf_magnitude_baseline[:,:,None]
    return res

def calc_t_stat(target_data, nontarget_data):
    res = np.zeros((target_data.shape[1],target_data.shape[2],target_data.shape[3]))
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

if __name__=='__main__':
    exp_num=sys.argv[1]
    path = join('..', 'meg_data1',exp_num)

    #Loading data
    #data start time = -820
    target_grad_data, nontarget_grad_data = get_data(path,'MEG GRAD') #trials x channels x times
    first_grad_target = tft_transofrm(target_grad_data) # trials x channels x freqs x times
    first_grad_nontarget = tft_transofrm(nontarget_grad_data)

    second_grad_target = baseline_correction(first_grad_target)
    second_grad_nontarget = baseline_correction(first_grad_nontarget)

    # Calc mean for UNCORRECTED data
    third_grad_target = first_grad_target.mean(axis=0)
    third_grad_nontarget = first_grad_nontarget.mean(axis=0)
    save_mat(third_grad_target,'third_grad_target')
    save_mat(third_grad_nontarget,'third_grad_nontarget')
    vis_each_freq(third_grad_target,'grad_mean_target_notcorrected')
    vis_each_freq(third_grad_nontarget,'grad_mean_nontarget_notcorrected')

    # Calc mean for CORRECTED data
    fourth_grad_target = second_grad_target.mean(axis=0)
    fourth_grad_nontarget = second_grad_nontarget.mean(axis=0)
    save_mat(fourth_grad_target,'fourth_grad_target')
    save_mat(fourth_grad_nontarget,'fourth_grad_nontarget')
    vis_each_freq(fourth_grad_target,'grad_mean_target_corrected')
    vis_each_freq(fourth_grad_nontarget,'grad_mean_nontarget_corrected')

    # Calc t-stat for UNCORRECTED data
    fivth = ttest_ind(first_grad_target,first_grad_nontarget,axis=0,equal_var=False)
    vis_each_freq(fivth.statistic,'grad_t-stat_notcorrected')
    save_mat(fivth.statistic,'fivth')

    # Calc t-stat for CORRECTED data
    sixth = ttest_ind(second_grad_target,second_grad_nontarget,axis=0,equal_var=False)
    vis_each_freq(sixth.statistic,'grad_t-stat_corrected')
    save_mat(sixth.statistic,'sixth')


    target_mag_data, nontarget_mag_data = get_data(path,'MEG MAG')
    first_mag_target = tft_transofrm(target_mag_data) # trials x channels x freqs x times
    first_mag_nontarget = tft_transofrm(nontarget_mag_data)

    second_mag_target = baseline_correction(first_mag_target)
    second_mag_nontarget = baseline_correction(first_mag_nontarget)

    # Calc mean for UNCORRECTED data
    third_mag_target = first_mag_target.mean(axis=0)
    third_mag_nontarget = first_mag_nontarget.mean(axis=0)
    save_mat(third_mag_target,'third_mag_target')
    save_mat(third_mag_nontarget,'third_mag_nontarget')
    vis_each_freq(third_mag_target,'mag_mean_target_notcorrected')
    vis_each_freq(third_mag_nontarget,'mag_mean_nontarget_notcorrected')

    # Calc mean for CORRECTED data
    fourth_mag_target = second_mag_target.mean(axis=0)
    fourth_mag_nontarget = second_mag_nontarget.mean(axis=0)
    save_mat(fourth_mag_target,'fourth_mag_target')
    save_mat(fourth_mag_nontarget,'fourth_mag_nontarget')
    vis_each_freq(fourth_mag_target,'mag_mean_target_corrected')
    vis_each_freq(fourth_mag_nontarget,'mag_mean_nontarget_corrected')

    # Calc t-stat for UNCORRECTED data
    fivth = ttest_ind(first_mag_target,first_mag_nontarget,axis=0,equal_var=False)
    vis_each_freq(fivth.statistic,'mag_t-stat_notcorrected')
    save_mat(fivth.statistic,'fivth')

    # Calc t-stat for CORRECTED data
    sixth = ttest_ind(second_mag_target,second_mag_nontarget,axis=0,equal_var=False)
    vis_each_freq(sixth.statistic,'mag_t-stat_corrected')
    save_mat(sixth.statistic,'sixth')