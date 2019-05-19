#encoding:utf-8
import os
import cv2
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np 

out_path = '/media/ysliu/22AE2337AE230341/dlw_mat_file/python_tools/'

def makeDir(path):
    try:
        if not os.path.exists(path):
            if not os.path.isfile(path):
                # os.mkdir(path)
                os.makedirs(path)
            return 0
        else:
            return 1
    except Exception, e:
        print str(e)
        return -2
def save_cv2_path(out_path,img_list,label,img):
    sub_class = img_list.split('/')[-2]
    makeDir(sub_class)
    #print os.path.join(out_path+sub_class+'/'+img_list.split('/')[-1].split('.')[0]+'_'+str(label)+'.jpg')
    #print img_list.split('/')[-1].split('.')[0]
    print "satrt!!!!"
    path1 = os.path.join(out_path+sub_class+'/'+img_list.split('/')[-1].split('.')[0]+'_'+str(label)+'.jpg')
    print path1
    cv2.imwrite(path1,img)
    print "end!!!!"
def save_Image_path(out_path,img_list,label,img):
    sub_class = img_list.split('/')[-2]
    makeDir(sub_class)
    #print os.path.join(out_path+sub_class+'/'+img_list.split('/')[-1].split('.')[0]+'_'+str(label)+'.jpg')
    #print img_list.split('/')[-1].split('.')[0]
    print "satrt!!!!"
    path1 = os.path.join(out_path+sub_class+'/'+img_list.split('/')[-1].split('.')[0]+'_'+str(label)+'.jpg')
    print path1
    img.save(path1)
    print "end!!!!"
   




#crop_corner
def img_crop50(img_list,crop_size):#182
    image = cv2.imread(img_list)
    #save redion image
    #save_cv2_path(out_path,img_list,'img_region',image)
    (height,weight)= image.shape[:2]
    crop_size = crop_size
    size = (224,224)
    #left_top
    img_left_top50 = cv2.resize(image[0:crop_size,0:crop_size],size)
    save_cv2_path(out_path,img_list,'img_left_top50',img_left_top50)
    #right_top 
    img_right_top50 = cv2.resize(image[74:weight,0:crop_size],size)
    save_cv2_path(out_path,img_list,'img_right_top50',img_right_top50)
    #left_bottom
    img_left_bottom50 = cv2.resize(image[0:crop_size,74:height],size)
    save_cv2_path(out_path,img_list,'img_left_bottom50',img_left_bottom50)
    #right_bottom
    img_right_bottom50 = cv2.resize(image[74:weight,74:height],size)
    save_cv2_path(out_path,img_list,'img_right_bottom50',img_right_bottom50)
    #center_area
    img_center50 = cv2.resize(image[37:219,37:219],size)
    save_cv2_path(out_path,img_list,'img_center50',img_center50)
def img_crop75(img_list,crop_size):#222
    image = cv2.imread(img_list)
    #save redion image
    #save_cv2_path(out_path,img_list,'img_region',image)
    (height,weight)= image.shape[:2]
    crop_size = crop_size
    size = (224,224)
    #left_top
    img_left_top75 = cv2.resize(image[0:crop_size,0:crop_size],size)
    save_cv2_path(out_path,img_list,'img_left_top75',img_left_top75)
    #right_top 
    img_right_top75 = cv2.resize(image[34:weight,0:crop_size],size)
    save_cv2_path(out_path,img_list,'img_right_top75',img_right_top75)
    #left_bottom
    img_left_bottom75 = cv2.resize(image[0:crop_size,34:height],size)
    save_cv2_path(out_path,img_list,'img_left_bottom75',img_left_bottom75)
    #right_bottom
    img_right_bottom75 = cv2.resize(image[34:weight,34:height],size)
    save_cv2_path(out_path,img_list,'img_right_bottom75',img_right_bottom75)
    #center_area
    img_center75 = cv2.resize(image[17:239,17:239],size)
    save_cv2_path(out_path,img_list,'img_center75',img_center75)

#img_crop('/media/dlw/work/SBS-CNN/alexnet4_siamese/datasets4/airplane/airplane_001.jpg',227)
#flip_horization
def img_flip(img_list,use_cv2=False):
    image = Image.open(img_list)
    img_resize = image.resize((224,224),Image.BILINEAR)
    save_Image_path(out_path,img_list,'img_resize',img_resize)
    #img_filp_top_bottom
    img_flip_top_bottom = image.transpose(Image.FLIP_TOP_BOTTOM).resize((224,224),Image.BILINEAR)
    save_Image_path(out_path,img_list,'img_flip_top_bottom',img_flip_top_bottom)

    img_flip_left_right = image.transpose(Image.FLIP_LEFT_RIGHT).resize((224,224),Image.BILINEAR)
    save_Image_path(out_path,img_list,'img_flip_left_right',img_flip_left_right)

    img_rotate_90 = image.transpose(Image.ROTATE_90).resize((224,224),Image.BILINEAR)
    save_Image_path(out_path,img_list,'img_rotate_90',img_rotate_90)
   
    img_rotate_180 = image.transpose(Image.ROTATE_180).resize((224,224),Image.BILINEAR)
    #save_Image_path(out_path,img_list,'img_rotate_180',img_rotate_180)

    img_rotate_270 = image.transpose(Image.ROTATE_270).resize((224,224),Image.BILINEAR)
    save_Image_path(out_path,img_list,'img_rotate_270',img_rotate_270)

#img_flip('/media/dlw/work/SBS-CNN/alexnet4_siamese/datasets4/airplane/airplane_001.jpg')

def randomColor(img_list):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    image = Image.open(img_list)
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    #save_Image_path(out_path,img_list,'color_image',color_image)

    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    #save_Image_path(out_path,img_list,'brightness_image',brightness_image)

    random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor) # 调整图像对比度
    #save_Image_path(out_path,img_list,'contrast_image',contrast_image)

    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    img_sharpness = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
    save_Image_path(out_path,img_list,'img_sharpness',img_sharpness)

#randomColor('/media/dlw/work/SBS-CNN/alexnet4_siamese/datasets4/airplane/airplane_001.jpg')


#rotate 
def rotate(img_list,center=None,scale=1.0):
	image = cv2.imread(img_list)
	(h,w) = image.shape[:2]
	# center is not specific 
	if center is None:
		center = (h/2,w/2)
	M = cv2.getRotationMatrix2D(center,30,scale)
	rotated = cv2.warpAffine(image,M,(h,w))

	#save_cv2_path(out_path,img_list,'rotate_30',rotated)
if __name__ == '__main__':
    height = 227
    width = 227
    dir_path = '/media/ysliu/22AE2337AE230341/dlw_mat_file/data45'
    for dir_root,sub_class,file_names in os.walk(dir_path):
        for file_name in file_names:
            img_list = os.path.join(dir_root+'/'+file_name)
            img_crop50(img_list,182)
            img_crop75(img_list,222)
            #rotate(img_list)
            #randomColor(img_list)
            img_flip(img_list)
            #img_crop(img_list,227)

# -*- coding:utf-8 -*-
"""数据增强
   1. 翻转变换 flip
   2. 随机修剪 random crop
   3. 色彩抖动 color jittering
   4. 平移变换 shift
   5. 尺度变换 scale
   6. 对比度变换 contrast
   7. 噪声扰动 noise
   8. 旋转变换/反射变换 Rotation/reflection
"""
 
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import threading, os, time
import logging
 
logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True
 
 
class DataAugmentation:
    """
    包含数据增强的八种方式
    """
    def __init__(self):
        pass
 
    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")
 
    @staticmethod
    def randomRotation(image, mode=Image.BICUBIC):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode)
 
    @staticmethod
    def randomCrop(image):
        """
        对图像随意剪切,考虑到图像大小范围(68,68),使用一个一个大于(36*36)的窗口进行截图
        :param image: PIL的图像image
        :return: 剪切之后的图像
        """
        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_size = np.random.randint(40, 48)
        random_region = (
            (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
            (image_height + crop_win_size) >> 1)
        return image.crop(random_region)
 
    @staticmethod
    def randomColor(image):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
 
    @staticmethod
    def randomGaussian(image, mean=0.2, sigma=0.3):
        """
         对图像进行高斯噪声处理
        :param image:
        :return:
        """
       #
 
        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im
 
        # 将图像转化成数组
        img = np.asarray(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img))
 
    @staticmethod
    def saveImage(image, path):
        image.save(path)
 
 
def makeDir(path):
    try:
        if not os.path.exists(path):
            if not os.path.isfile(path):
                # os.mkdir(path)
                os.makedirs(path)
            return 0
        else:
            return 1
    except Exception, e:
        print str(e)
        return -2
 
 
def imageOps(func_name, image, des_path, file_name, times=5):
    funcMap = {"randomRotation": DataAugmentation.randomRotation,
               "randomCrop": DataAugmentation.randomCrop,
               "randomColor": DataAugmentation.randomColor,
               "randomGaussian": DataAugmentation.randomGaussian
               }
    if funcMap.get(func_name) is None:
        logger.error("%s is not exist", func_name)
        return -1
 
    for _i in range(0, times, 1):
        new_image = funcMap[func_name](image)
        DataAugmentation.saveImage(new_image, os.path.join(des_path, func_name + str(_i) + file_name))
 
 
opsList = {"randomRotation", "randomCrop", "randomColor", "randomGaussian"}
 
 
def threadOPS(path, new_path):
    """
    多线程处理事务
    :param src_path: 资源文件
    :param des_path: 目的地文件
    :return:
    """
    if os.path.isdir(path):
        img_names = os.listdir(path)
        
    else:
        img_names = [path]
    for img_name in img_names:
        print img_name
        tmp_img_name = os.path.join(path, img_name)
        if os.path.isdir(tmp_img_name):
            if makeDir(os.path.join(new_path, img_name)) != -1:
                threadOPS(tmp_img_name, os.path.join(new_path, img_name))
            else:
                print 'create new dir failure'
                return -1
                # os.removedirs(tmp_img_name)
        elif tmp_img_name.split('.')[1] != "DS_Store":
            # 读取文件并进行操作
            image = DataAugmentation.openImage(tmp_img_name)
            threadImage = [0] * 5
            _index = 0
            for ops_name in opsList:
                threadImage[_index] = threading.Thread(target=imageOps,
                                                       args=(ops_name, image, new_path, img_name,))
                threadImage[_index].start()
                _index += 1
                time.sleep(0.2)
 
 
#if __name__ == '__main__':
    #threadOPS("/media/dlw/work/SBS-CNN/alexnet4_siamese/datasets4/airplane/",
     #         "/media/dlw/work/SBS-CNN/alexnet4_siamese/12306train3/")
