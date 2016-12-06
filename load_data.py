from scipy.io import loadmat
from os import listdir
from os.path import join
import numpy as np

def load_data(path,sensor_type):
    # sensor_type -  'MEG GRAD' or 'MEG MAG'
    mask_rawdata = loadmat('ChannelType.mat')
    if sensor_type == 'EOG':
        sensor_mask = np.full((len(mask_rawdata['Type'])),False,dtype=bool)
        sensor_mask[315] = True # 316,317 - EOG channel indices
        sensor_mask[316] = True
    else:
        sensor_mask = np.array(map(lambda x: x[0][0] == sensor_type,mask_rawdata['Type']))

    return np.concatenate([extract_grad_mat(join(path,f),sensor_mask) for f in listdir(path) if f.endswith(".mat")],axis=0)


def extract_grad_mat(path,gradiom_mask):
    data=loadmat(path)
    return (data['F'][gradiom_mask])[np.newaxis,...].astype('float32',casting='same_kind') #additional dimension for easier concatenation to 3d array in the future

def get_data(path,sensor_type):
    # sensor_type -  'MEG GRAD' or 'MEG MAG'
    # path_to_target = join(path, 'BallChosenSI')
    path_to_nontarget = join(path, 'ErrorBallChosen')
    # target_data = load_data(path_to_target,sensor_type) # trials x time x channel
    nontarget_data = load_data(path_to_nontarget,sensor_type)
    # return target_data, nontarget_data
    return nontarget_data