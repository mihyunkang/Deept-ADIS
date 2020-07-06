import os
from random import choice
import shutil

#arrays to store file names
imgs =[]

#setup dir names
trainPath = './face_data_div/train/real_15/'
valPath = './face_data_div/val/real_15/'
crsPath = './real_per_15_frame/' #dir where images and annotations stored

#setup ratio (val ratio = rest of the files in origin dir after splitting into train and test)
train_ratio = 0.9
val_ratio = 0.1


#total count of imgs
totalImgCount = len(os.listdir(crsPath))

#soring files to corresponding arrays
for (dirname, dirs, files) in os.walk(crsPath):
    for filename in files:
        imgs.append(filename)


#counting range for cycles
countForTrain = int(len(imgs)*train_ratio)
countForVal = int(len(imgs)*val_ratio)

#cycle for train dir
for x in range(countForTrain):

    fileJpg = choice(imgs) # get name of random image from origin dir

    #move both files into train dir
    shutil.move(os.path.join(crsPath, fileJpg), os.path.join(trainPath, fileJpg))

    #remove files from arrays
    imgs.remove(fileJpg)



#cycle for test dir   
for x in range(countForVal):

    fileJpg = choice(imgs) # get name of random image from origin dir

    #move both files into train dir
    shutil.move(os.path.join(crsPath, fileJpg), os.path.join(valPath, fileJpg))

    #remove files from arrays
    imgs.remove(fileJpg)

#rest of files will be validation files, so rename origin dir to val dir
#os.rename(crsPath, valPath)

#summary information after splitting
print('Total images: ', totalImgCount)
print('Images in train dir:', len(os.listdir(trainPath)))
print('Images in validation dir:', len(os.listdir(valPath)))