from load_data import get_data
import sys
from mne.time_frequency import cwt_morlet
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import os
from scipy.io import savemat,loadmat

from vis_heads import vis_heads_array
from statsmodels.sandbox.stats.multicomp import fdrcorrection0
from scipy.signal import butter, lfilter



def tft_transofrm(source,freqs):
    # average : bool - averaging of output data in sliding windows
    # average_w_width sliding window width
    # average_w_step sliding window step

    sfreq = 1000
    res = np.zeros((source.shape[0],source.shape[1],len(freqs),source.shape[2]),dtype=np.float32)
    for i in xrange(source.shape[0]):
         res[i,:,:,:] = np.absolute(cwt_morlet(source[i,:,:], sfreq, freqs, use_fft=True, n_cycles=7.0, zero_mean=True, decim=1)).astype('float32',casting='same_kind')
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


def save_results(data,mask,freqs,sensor_type,title,result_path):
    # mat_path = os.path.join(result_path,'mat')
    # rect_img_path = os.path.join(result_path, 'rectangle_images')
    heads_img_path = os.path.join(result_path, 'heads_images')
    # if not os.path.isdir(mat_path): os.makedirs(mat_path)
    # if not os.path.isdir(rect_img_path): os.makedirs(rect_img_path)
    if not os.path.isdir(heads_img_path): os.makedirs(heads_img_path)
    # savemat(file_name = os.path.join(mat_path,title),mdict=dict(data=data)) #save data as .mat file with 'data' variable inside [channelas x freqs x times]
    # fig = vis_space_freq(data, title, freqs)
    # plt.savefig(os.path.join(rect_img_path, title  + '.png'))
    # plt.close(fig)

    heads = [('FDR NOT passed=%d, %.02f Hz' % (all(mask[:,fq_ind]==False)==True,freqs[fq_ind]), data[:, fq_ind], mask[:, fq_ind]) for fq_ind,_ in enumerate(freqs)]
    fig = vis_heads_array('%s' % title, sensor_type.lower(), *heads)
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


def fdr_correction(p_values,thres):
    mask, adjusted_p = fdrcorrection0(p_values.flatten(), thres)
    mask = mask.reshape(p_values.shape)
    return mask

def calc_metrics(data_path,left_border,result_path,exp_num,sensor_type,freqs):
    erase_dir(result_path)
    data = get_data(data_path,sensor_type) #trials x channels x times
    sensor_type = sensor_type.split(' ')[-1]

    data = get_tft_data(data,'target',data_path,sensor_type,freqs,save_tft=False,load_existing_tft=False) # trials x channels x freqs x times

    win_length = 300 #ms
    right_border = 200 #ms
    right_border_ind = right_border - left_border

    # np.seterr(all='raise')
    t_val = np.empty((data.shape[1],data.shape[2],0),np.float32)
    p_val = np.empty((data.shape[1], data.shape[2], 0), np.float32)
    times = np.arange(0,right_border_ind-win_length+1,win_length)
    for ind,t in enumerate(times):

        loop_t_val,loop_p_val = ttest_ind(data[:,:,:,t:t+win_length].mean(axis=3),
                                          data[:,:,:,right_border_ind:right_border_ind+win_length].mean(axis=3) ,
                                          axis=0,equal_var=True)
        t_val = np.concatenate((t_val,loop_t_val[:,:,np.newaxis]),axis=2)
        p_val = np.concatenate((p_val, loop_p_val[:, :, np.newaxis]), axis=2)

    fdr_thres = 0.3
    t_mask = fdr_correction(p_val,fdr_thres) #channels x freqs x times
    t_fdr = max(t_val[t_mask].min(),t_val[t_mask].min(),key=abs)
    for fq_ind,fq in enumerate(freqs):
        title = '%4.1f Hz %s t_abs=%d, fdr_thres=%0.02f, FDRpassed t=%0.02f' %(fq,exp_num,max(abs(t_val[:,fq_ind,:].min()), abs(t_val[:,fq_ind,:].max())),fdr_thres,t_fdr)
        save_results(t_val[:,fq_ind,:],t_mask[:,fq_ind,:],(times+left_border),sensor_type, title, result_path)



def t_stat_target_nontarget(data_path,left_border,result_path,exp_num,sensor_type,freqs):
    target_data,nontarget_data = get_data(data_path, sensor_type)  # trials x channels x times
    sensor_type = sensor_type.split(' ')[-1]
    target_data = get_tft_data(target_data, 'target', data_path, sensor_type, freqs, save_tft=False,
                        load_existing_tft=False)  # trials x channels x freqs x times
    nontarget_data = get_tft_data(nontarget_data, 'target', data_path, sensor_type, freqs, save_tft=False,
                               load_existing_tft=False)  # trials x channels x freqs x times
    fix_window_begin = 200-left_border
    fix_window_end = 500-left_border
    t_val, p_val = ttest_ind(target_data[:, :, :, fix_window_begin:fix_window_end].mean(axis=3),
                             nontarget_data[:, :, :, fix_window_begin:fix_window_end].mean(axis=3),
                                      axis=0, equal_var=True)
    fdr_thres = 0.3
    t_mask = fdr_correction(p_val, fdr_thres)  # channels x freqs x times

    if t_val[t_mask].size !=0:
        t_fdr = max(t_val[t_mask].min(), t_val[t_mask].min(), key=abs)
    else:
        t_fdr =0.0001

    title = '%s t_abs=%d, fdr_thres=%0.02f, FDRpassed t=%0.02f' % (exp_num, max(abs(t_val.min()), abs(t_val.max())), fdr_thres, t_fdr)
    save_results(t_val, t_mask, freqs, sensor_type, title, result_path)

def test_time_windows(data_path,data_left_border,result_path,exp_num,sensor_type):
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = np.zeros(data.shape)
        for tr in range(data.shape[0]):
            for ch in range(data.shape[1]):
                y[tr, ch, :] = lfilter(b, a, data[tr, ch, :])
        return y

    def baseline_correction(data, epoch_start_time,baseline_start = 200,baseline_end = 500): #baseline times independent from window borders
        start_base_index = max((baseline_start - epoch_start_time), 0)
        end_base_index = baseline_end - epoch_start_time
        # res = np.zeros(data.shape,dtype=np.float32)
        bline = data[:, :, start_base_index:end_base_index].mean(axis=2)
        return data - bline[:,:,None]

    if not os.path.isdir(result_path): os.makedirs(result_path)
    data = get_data(data_path, sensor_type)  # trials x channels x times
    sensor_type = sensor_type.split(' ')[-1]
    data = butter_lowpass_filter(data, cutoff=15, fs=1000, order=5)
    if (sensor_type == 'EOG'):
        ch_inds = [0,1]
    for ch in ch_inds:
        plt.plot(np.arange(-500, 501), data[:, ch, -500+data_left_border:].transpose().mean(axis=1))
        title = 'MEAN_Ch=%d_%s_%s' % (ch, sensor_type, exp_num)
        plt.title(title)
        plt.savefig(os.path.join(result_path, title + '.png'))
        plt.close()

        plt.plot(np.arange(-500, 501), (baseline_correction(data, data_left_border)[:,ch,-500+data_left_border:]).transpose())
        title = 'Ch=%d_%s_%s' %(ch,sensor_type,exp_num)
        plt.title(title)
        plt.savefig(os.path.join(result_path,title +'.png'))
        plt.close()



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
        freqs = range(10,12,1)
        res_dir = 'tmp'
        erase_dir(res_dir)
    else:
        freqs = range(10,100,5)
        res_dir = 'results'

    data_left_border = -4300
    # result_path = os.path.join(res_dir,'butterfly')
    # test_time_windows(data_path,data_left_border,result_path,exp_num,'EOG')
    result_path = os.path.join(res_dir, exp_num, 'MAG', 'target_vs_nontarget')
    t_stat_target_nontarget(data_path, data_left_border, result_path, exp_num, 'MEG MAG', freqs)

    # result_path = os.path.join(res_dir,exp_num,'GRAD','fix_vs_nonfix')
    # calc_metrics(data_path,data_left_border,result_path,exp_num,'MEG GRAD',freqs)
    #
    # result_path = os.path.join(res_dir,exp_num,'MAG','fix_vs_nonfix')
    # calc_metrics(data_path,data_left_border,result_path,exp_num,'MEG MAG',freqs)
