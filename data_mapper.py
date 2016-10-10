import numpy as np
from os import path
import os

class DataMapper:
    def __init__(self,filename):
        self.filename = filename
        # self.tmp_dir_path = mkdtemp()
        pass
    def store_data(self,data,num_of_chunks):
        #Method devids data on 'num_of_chunks'
        #returns file descriptor on file, which contain n-1 chunks, and one chunk of data
        filename = path.join(self.filename)

        channel_num = data.shape[1]
        self.size_of_chunk = (channel_num / num_of_chunks)
        self.avail_chunks = num_of_chunks - 1 #because we return first chunk in this method
        self.total_chunks = self.avail_chunks
        res_chunk_chan_num = (channel_num / num_of_chunks)  + channel_num % num_of_chunks
        chunk_shape = (data.shape[0],res_chunk_chan_num,data.shape[2]) #shape of data wich returned from function
        flushed_chunk_shape = (data.shape[0],channel_num  - res_chunk_chan_num,data.shape[0]) #shape of data wich flushed to disk

        self.fp = np.memmap(filename, dtype='float32', mode='w+', shape=flushed_chunk_shape)
        self.fp[:] = data[:,res_chunk_chan_num:,:]
        self.fp.flush()
        ret_chunk = np.zeros(chunk_shape,dtype='float32')
        ret_chunk[:] = data[:,0:res_chunk_chan_num,:]
        return ret_chunk

    def get_next_chunk(self,need_chunks):
        chunks_will_return=0
        if self.avail_chunks > 0:
            if type(need_chunks) is str:
                if need_chunks == 'all':
                    chunks_will_return = self.avail_chunks
            if type(need_chunks) is int:
                    if need_chunks <= self.avail_chunks:
                       chunks_will_return = need_chunks
                    else:
                        print 'avail chunks less then requested'
                        return
            start = (self.total_chunks - self.avail_chunks)
            end = start + chunks_will_return*self.size_of_chunk
            res = self.fp[:,start:end,:]
            # self.fp[:]=np.delete(self.fp,range(chunks_will_return*self.size_of_chunk),axis=1)
            self.avail_chunks -= chunks_will_return
            # self.fp.flush()
            return res
        else:
            print 'File is empty'
            return

    def _clean_dir(self):
        # for root, dirs, files in os.walk(self.tmp_dir_path, topdown=False):
        #     for name in files:
        #         os.remove(os.path.join(root, name))
        #     for name in dirs:
        #         os.rmdir(os.path.join(root, name))
        # os.rmdir(self.tmp_dir_path)
        os.remove(self.filename)

    def __close__(self):
        del self.fp
        os.remove(self.filename)


    def __del__(self):
        del self.fp
        os.remove(self.filename)




