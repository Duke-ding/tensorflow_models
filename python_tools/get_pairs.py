#coding:utf-8
import os
import random
import numpy as np 

#设置随机数种子
random.seed(1)

def save_txt(save_path,img_path):
	f = open(save_path,'a')
	f.write(img_path+'\n')
	f.close()


dir_root = '/media/dlw/work/SBS-CNN/data45_train/'
save_path1 = '/media/dlw/work/python_test/path/train26_01.txt'
save_path2 = '/media/dlw/work/python_test/path/train26_label.txt'
#
class_id = os.listdir(dir_root)#type list
print class_id
class_label = [x for x in range(45)]
#print(type(class_label))
class_label_id = dict(zip(class_id,class_label))





for i in range(63000):
	class_num1 = random.randint(0,44)
	class_num2 = random.randint(0,44)
	#image_list
	img_file1 = os.listdir(os.path.join(dir_root+class_id[class_num1]))
	print(len(img_file1))
	#print(img_file)
	img_num1 = random.randint(0,139)
	img_path1 = os.path.join(os.path.join(class_id[class_num1])+'/'+img_file1[img_num1]+' '+str(1))
	img_num2 = random.randint(0,139)
	img_path2 = os.path.join(os.path.join(class_id[class_num1])+'/'+img_file1[img_num2]+' ' +str(class_label_id[class_id[class_num1]]))
	save_txt(save_path1,img_path1)
	save_txt(save_path2,img_path2)

	img_file2 = os.listdir(os.path.join(dir_root+class_id[class_num2]))
	#print(img_file)
	img_num1 = random.randint(0,139)
	img_path1 = os.path.join(os.path.join(class_id[class_num1])+'/'+img_file1[img_num1]+' '+str(0))
	img_num2 = random.randint(0,139)
	img_path2 = os.path.join(os.path.join(class_id[class_num2])+'/'+img_file2[img_num2]+' ' +str(class_label_id[class_id[class_num1]]))
	save_txt(save_path1,img_path1)
	save_txt(save_path2,img_path2)
	print("开始保存第%s张图片！"%str(i))






"""
#图像文件路径列表
data_path = []
for data_root,sub_root,file_name in os.walk(image_root):
	for filenames in file_name:
		img_path = os.path.join(data_root.split('/')[-1]+'/'+filenames)
		data_path.append(img_path)

print(data_path[:140])
#随机数
num = len(data_path[:140])
num1= len(data_path)/45
print num1
#创建图像路径对
for i in range(1):
	image_path1 = data_path[random.randint(0,num-1)]
	#print(image_path1)
	image_path2 = data_path[random.randint(0,num-1)]
	#print(image_path2)
	img1 = image_path1.split('/')[0]
	print(img1)
	img2 = image_path2.split('/')[0]
	print(img2)
	if img1 == img2:
		print('相似！')
	else:
		print('不相似！')

"""
#得到svm_path_路径
import os
import random
from contextlib import nested
print('------开始生成训练数据集标签！------')
data_dir = '/media/dlw/work/SBS-CNN/data45_train'
data_path = '/media/dlw/work/SBS_classification/path/path_svm/train_svm_path.txt'
label_path = '/media/dlw/work/SBS_classification/path/path_svm/train_svm_label.txt'

def create_label(file_path,file_label,file_label1):
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
            data_path.append(file_root2)
    print('image_nums:',len(data_path))
    random.shuffle(data_path) 
    with nested(open(file_label,'w'),open(file_label1,'w')) as (filewrite,filewrite1):
        for file1 in data_path:
			filewrite.write(file1)
			filewrite.write('\n')
			#读取label
			img_label=file1.split(' ')[-1]
			filewrite1.write(img_label)
			filewrite1.write('\n')


create_label(data_dir,data_path,label_path)














"""
