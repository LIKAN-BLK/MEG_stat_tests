from scipy.io import loadmat
from os import listdir
from os.path import join
import numpy as np



def load_data(path,sensor_type):
    # sensor_type -  'MEG GRAD' or 'MEG MAG'
    mask_rawdata = loadmat('ChannelType.mat')
    gradiom_mask = np.array(map(lambda x: x[0][0] == sensor_type,mask_rawdata['Type']))
    return np.concatenate([extract_grad_mat(join(path,f),gradiom_mask) for f in listdir(path) if f.endswith(".mat")],axis=0)


def extract_grad_mat(path,gradiom_mask):
    data=loadmat(path)
    return (data['F'][gradiom_mask])[np.newaxis,...].astype('float32',casting='same_kind') #additional dimension for easier concatenation to 3d array in the future

def get_data(path,sensor_type):
    # sensor_type -  'MEG GRAD' or 'MEG MAG'
    target_data = load_data(path,sensor_type) # trials x time x channel
    return target_data
if __name__== '__main__':
    print('It\'s fun!')