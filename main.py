from load_data import get_data
import sys
from os.path import join
from mne.time_frequency import cwt_morlet
import numpy as np
from scipy.stats import ttest_ind

def tft_transofrm(source):
    window_start = 820 #100ms after fuxation
    window_end = window_start+400


    freqs = range(10,100,1)
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
    for i in xrange(data.shape[0]):
        tf_magnitude = data[i,-1]
        tf_magnitude_baseline = np.log10(data[i,:,:,baseline_start:baseline_end].mean(axis=2))
        res[i,:,:,:] = np.log10(data[i,:,:,:]) - tf_magnitude_baseline[:,:,None]
    return res

def calc_t_stat(target_data, nontarget_data):
    res = np.zeros((target_data.shape[1],target_data.shape[2],target_data.shape[3]))
    res = ttest_ind(target_data,nontarget_data,axis=0,equal_var=False)
    return res

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

    third_grad_target = second_grad_target.mean(axis=0)
    third_grad_nontarget = second_grad_nontarget.mean(axis=0)

    fourth_grad_target = third_grad_target.mean(axis=0)
    fourth_grad_nontarget = third_grad_nontarget.mean(axis=0)

    fivth = ttest_ind(first_grad_target,first_grad_nontarget,axis=0,equal_var=False)
    sixth = ttest_ind(second_grad_target,second_grad_nontarget,axis=0,equal_var=False)


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