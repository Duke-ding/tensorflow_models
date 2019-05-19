#coding:utf-8
import os
import glob 


import PIL
from PIL import Image
import os
import sys
def readf(input_dir,img_size,output_dir):
    try:
        print "starting...."
        print "Colecting data from %s " % input_dir
        tclass = [ d for d in os.listdir( input_dir ) ]
        counter = 0
        strdc = ''
        hasil = []
        for x in tclass:
           list_dir =  os.path.join(input_dir, x )
           list_tuj = os.path.join(output_dir+'/', x+'/')
           if not os.path.exists(list_tuj):
                os.makedirs(list_tuj)
           if os.path.exists(list_tuj):
               for d in os.listdir(list_dir):
                   try:
                       img = Image.open(os.path.join(input_dir+'/'+x,d))
                       img = img.resize((int(img_size),int(img_size)),Image.ANTIALIAS)
                       fname,extension = os.path.splitext(d)
                       newfile = fname+extension
                       if extension != ".jpg" :
                           newfile = fname + ".jpg"
                       img.save(os.path.join(output_dir+'/'+x,newfile),"JPEG",quality=90)
                       print "Resizing file : %s - %s " % (x,d)
                   except Exception,e:
                        print "Error resize file : %s - %s " % (x,d)
                        sys.exit(1)
               counter +=1
    except Exception,e:
        print "Error, check Input directory etc : ", e
        sys.exit(1)
        
        
#训练集
train_dir='/media/dlw/work/SBS-CNN/data45_train/'
img_size=227
train_output_dir='/media/dlw/work/python_test/train227_45'
readf(train_dir,img_size,train_output_dir)

#训练集
test_dir='/media/dlw/work/SBS-CNN/data45_test/'
img_size=227
test_output_dir='/media/dlw/work/python_test/test227_45'
readf(test_dir,img_size,test_output_dir)
