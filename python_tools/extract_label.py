#encoding:utf-8
#读取数据原始标签并得到提取lmdb格式的数据文件格式 data_root 
import os 
from contextlib import nested
import random
def extract_label(data_path1,save_path1):
	with nested(open(data_path1),open(save_path1,'w')) as (f1,f2):
		for file_path in f1.readlines():
			path_root = file_path.split(' ')
			#label = class_names_to_ids[str(path_root[0].split('/')[-2])]
			#print (str(label)+'\n')
			#l.append(label)
			label = path_root[-1]
			f2.writelines(str(label))
			#d.append(dir_path)

train_data_path1 = "/media/dlw/work/featurefusion/data_path21/data21_test.txt"
train_svm_label = "/media/dlw/work/featurefusion/data_path21/query21_test_label.txt"




extract_label(train_data_path1,train_svm_label)
#extract_label(test_data_path1,test_svm_label)
