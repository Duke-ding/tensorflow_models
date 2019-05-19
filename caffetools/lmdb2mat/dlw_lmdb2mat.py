 # -*- coding: UTF-8 -*-
import lmdb
import feat_helper_pb2
import numpy as np
import scipy.io as sio
import time

def main():
    lmdb_name = "/media/dlw/work/featurefusion/bvlc_alexnet/feature/alexnet_myfc6_test21_4"
    print ("%s" % lmdb_name)
    batch_num = int(1);
    batch_size = int(2100);
    window_num = batch_num*batch_size;

    start = time.time()
    if 'db' not in locals().keys():
        db = lmdb.open(lmdb_name)
        txn= db.begin()
        cursor = txn.cursor()
        cursor.iternext()
        datum = feat_helper_pb2.Datum()

        keys = []
        values = []
        for key, value in enumerate( cursor.iternext_nodup()):
            keys.append(key)
            values.append(cursor.value())
        print(len(values))
    for i in range(batch_num):
        ft = np.zeros((batch_size, int(4096)))
        for im_idx in range(batch_size*i,batch_size*(i+1)):
        	datum.ParseFromString(values[im_idx])
        	ft[im_idx-(batch_size)*i,:] = datum.float_data

        print('deal with%s '% str(i))
        

        print ('time 1: %f' %(time.time() - start))
        sio.savemat("/media/dlw/work/featurefusion/bvlc_alexnet/feature/alexnet_myfc6_test21_4%s" % str(i) ,{'feats':ft})
        print ('time 2: %f' %(time.time() - start))
        print ('done!')

if __name__ == '__main__':
    import sys
    main()
