from load_data import get_data
import sys
from os.path import join
from mne.time_frequency import cwt_morlet
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import os

def tft_transofrm(source):
    window_start = 820 #100ms after fuxation
    window_end = window_start+400


    freqs = range(10,15,1)
    sfreq = 1000
    res = np.zeros((source.shape[0],source.shape[1],len(freqs),source.shape[2]))
    for i in xrange(source.shape[0]):
        tf_magnitude = np.absolute(cwt_morlet(source[i,:,:], sfreq, freqs, use_fft=True, n_cycles=7.0, zero_mean=True, decim=1))
        res[i,:,:,:] = (tf_magnitude)
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

    plt.title(title)
    plt.imshow(data,aspect='auto')
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Channel')

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
        visualise(data[:,fq,:],'%s fq=%f' %(title,fq))
        plt.savefig(os.path.join('results',title,'_fq=%0.1f.png' % fq))
        plt.close()


if __name__=='__main__':
    exp_num=sys.argv[1]
    path = join('..', 'meg_data1',exp_num)

    #Loading data
    #data_start_time = -820
    target_grad_data, nontarget_grad_data = get_data(path,'MEG GRAD') #trials x channels x times
    first_grad_target = tft_transofrm(target_grad_data) # trials x channels x freqs x times
    first_grad_nontarget = tft_transofrm(nontarget_grad_data)

    second_grad_target = baseline_correction(first_grad_target)
    second_grad_nontarget = baseline_correction(first_grad_nontarget)

    third_grad_target = first_grad_target.mean(axis=0)
    third_grad_nontarget = first_grad_nontarget.mean(axis=0)
    vis_each_freq(third_grad_target,'grad_mean_target_notcorrected')
    vis_each_freq(third_grad_nontarget,'grad_mean_nontarget_notcorrected')

    fourth_grad_target = third_grad_target.mean(axis=0)
    fourth_grad_nontarget = third_grad_nontarget.mean(axis=0)
    vis_each_freq(fourth_grad_target,'grad_mean_target_corrected')
    vis_each_freq(fourth_grad_nontarget,'grad_mean_nontarget_corrected')

    fivth = ttest_ind(first_grad_target,first_grad_nontarget,axis=0,equal_var=False)
    vis_each_freq(fivth,'grad_t-stat_notcorrected')
    sixth = ttest_ind(second_grad_target,second_grad_nontarget,axis=0,equal_var=False)
    vis_each_freq(sixth,'grad_t-stat_corrected')


    # target_mag_data, nontarget_mag_data = get_data(path,'MEG MAG')
    # first_mag_target = tft_transofrm(target_mag_data) # trials x channels x freqs x times
    # first_mag_nontarget = tft_transofrm(nontarget_mag_data)
    #
    #
    #
    # second_mag_target = baseline_correction(first_mag_target)
    # second_mag_nontarget = baseline_correction(first_mag_nontarget)
    #
    # third_mag_target = second_mag_target.mean(axis=0)
    # third_mag_nontarget = second_mag_nontarget.mean(axis=0)
    #
    # fourth_mag_target = third_mag_target.mean(axis=0)
    # fourth_mag_nontarget = third_mag_nontarget.mean(axis=0)
    #
    # fivth = ttest_ind(first_mag_target,first_mag_nontarget,axis=0,equal_var=False)
    # sixth = ttest_ind(second_mag_target,second_mag_nontarget,axis=0,equal_var=False)