#/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os 
import random
#训练数据集
print('------开始生成训练数据集标签！------')
data_dir = '/media/dlw/work/dlw_mat_file/data21'
data_path = '/media/dlw/work/featurefusion/data_path/data21.txt'

def create_label(file_path,file_label):
    classes = []
    labels = []
    data_path = []
    for data_root,sub_dir,filenames in os.walk(file_path):
        for label,sub_class in enumerate(sub_dir):
            labels.append(label)
            classes.append(sub_class)
    class_names_to_ids = dict(zip(classes,labels))
    for data_root,sub_dir,filenames in os.walk(file_path):
        for filename in filenames:
            file_root = os.path.join(data_root+'/'+filename)
            file_root1 = file_root.split('/')
            file_root2 = os.path.join(file_root1[-2]+'/'+file_root1[-1]+' '+ str(class_names_to_ids[file_root1[-2]]))
            #sub_classes = file_root.split('/')[-2]
            #classes_index = classes.index(sub_classes)
            #labels.append(int(classes_index))
            #file_path = os.path.join(file_root)
            data_path.append(file_root2)
            
            #print('开始读取数据：',file_path)
    print('image_nums:',len(data_path))
    #print('打乱数据前：')
    #print(data_path)
    random.shuffle(data_path)
    #print('打乱数据后')
    #print(data_path)
    #在一个列表中随机选择一部分数据
    #data_path1=random.sample(data_path,int(len(data_path)*0.1))
    with open(file_label,'w') as filewrite:
        for file in data_path:
			filewrite.write(file)
			filewrite.write('\n')


create_label(data_dir,data_path)

"""
import numpy as np
from sklearn.model_selection import train_test_split

filepath='/media/dlw/work/featurefusion/data_path/data1_path38.txt'  # 数据文件路径
data=np.loadtxt(filepath,dtype="str",delimiter=' ')

X, y = data[:,:-1],data[:,-1]

#【利用train_test_split方法，将X,y随机划分为训练集（X_train），训练集标签（y_train），测试集（X_test），测试集标签（y_test），按训练集：测试集=7:3的概率划分，到此步骤，可以直接对数据进行处理】
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y,shuffle=True)
print("train_nums:",len(X_train))
print("test_nums:",len(X_test))
#【将训练集与数据集的数据分别保存为CSV文件】
#np.column_stack将两个矩阵进行组合连接，numpy.savetxt 将txt文件保存为csv格式的文件
train= np.column_stack((X_train,y_train))
np.savetxt('/media/dlw/work/featurefusion/data_path/train_data38.txt',train, fmt='%s',delimiter = ' ')

test = np.column_stack((X_test, y_test))
np.savetxt('/media/dlw/work/featurefusion/data_path/test_data38.txt', test,fmt='%s', delimiter = ' ')

#X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.8, random_state=24,stratify=y,shuffle=True)

#【将训练集与数据集的数据分别保存为CSV文件】
#np.column_stack将两个矩阵进行组合连接，numpy.savetxt 将txt文件保存为csv格式的文件
#train1= np.column_stack((X_train1,y_train1))
#np.savetxt('/media/dlw/work/python_test/path/train_data_shuffle.txt',train1, fmt='%s',delimiter = ' ')

#test1 = np.column_stack((X_test1, y_test1))
#np.savetxt('/media/dlw/work/python_test/path/test_data_shuffle.txt', test1,fmt='%s', delimiter = ' ')
#####################################

import os 

from contextlib import nested
        
train_data_path1 ='/media/dlw/work/SBS-CNN/alexnet4_siamese/path/train_data.txt'
train_data_path2 ='/media/dlw/work/SBS-CNN/alexnet4_siamese/path/train_data_shuffle.txt'
train_save_path ='/media/dlw/work/SBS-CNN/alexnet4_siamese/path/train_data_shuffle_01.txt'

test_data_path1 = '/media/dlw/work/SBS-CNN/alexnet4_siamese/path/test_data.txt'
test_data_path2 = '/media/dlw/work/SBS-CNN/alexnet4_siamese/path/test_data_shuffle.txt'
test_save_path = '/media/dlw/work/SBS-CNN/alexnet4_siamese/path/test_data_shuffle_01.txt'

def add_0_1_label(data_path1,data_path2,save_path):
	with nested(open(data_path1),open(data_path2), open(save_path,'w')) as (f1,f2,f3):
		for f1_path,f2_path in zip(f1.readlines(),f2.readlines()):
			#print(f2_path)
			f1_label=f1_path.split(' ')[0].split('/')[-2]
			f2_label = f2_path.split('/')[-2]
			#print(f1_label,'=======',f2_label)
			if f1_label == f2_label:
				f3_path=os.path.join(f2_path.split(' ')[0]+' '+str(1)+'\n')#0不匹配
				f3.writelines(f3_path)
			else:
				f3_path=os.path.join(f2_path.split(' ')[0]+' '+str(0)+'\n')#1匹配
				f3.writelines(f3_path)
		print ('end!!!!!!')

#add_0_1_label(train_data_path1,train_data_path2,train_save_path)
#add_0_1_label(test_data_path1,test_data_path2,test_save_path)
import os
from contextlib import nested
data_dir = '/media/dlw/work/datasets/PatternNet_images/'
def remove_pathroot(data_path1,save_path1):
    with nested(open(data_path1),open(save_path1,'w')) as (f1,f2):
        for file_path in f1:
            f1_list = file_path.split(' ')
            f1_label_path = os.path.join(data_dir+f1_list[-2]+'\n')
            f2.writelines(f1_label_path)


train_data_path1="/media/dlw/work/SBS-CNN/data_path/train_data.txt"

train_save_path1="/media/dlw/work/SBS-CNN/data_path/train_data_path.txt"


test_data_path1="/media/dlw/work/SBS-CNN/data_path//test_data.txt"

test_save_path1="/media/dlw/work/SBS-CNN/data_path/test_data_path.txt"

#remove_pathroot(train_data_path1,train_save_path1)
#remove_pathroot(test_data_path1,test_save_path1)
"""
