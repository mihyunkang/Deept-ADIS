#train:test:val 폴더에 7:2:1 비율로 옮겨놓기.
#fake/real 데이터 각각!!
import os
import shutil
import random

##### 파일 불러오기
#real 데이터
path_real = "D:/data/full_data/real/"
#fake 데이터
path_fake = "D:/data/full_data/fake/" #path_fake = "D:/full_data/fake"
#파일경로로 불러옴
real_file = os.listdir(path_real)
#print(real_file[0:10])
fake_file = os.listdir(path_fake)
random.shuffle(real_file)
#print(real_file[0:10])
random.shuffle(fake_file)

##### 옮기는 경로~~~
#test 데이터
dest_test = "D:/data/full_data/test/"
#train 데이터
dest_train = "D:/data/full_data/train/"
#val 데이터
#dest_val = "D:/dataset/val/"

#real file - 2:8 분할
for i in range(len(real_file)):
    if(i%10==3 or i%10==7): #train-real
        shutil.copyfile(path_real+real_file[i], dest_test+'real/'+real_file[i])
    else: #val-real
        shutil.copyfile(path_real+real_file[i], dest_train+'real/'+real_file[i])

#fake file - 2:8 분할
for i in range(len(fake_file)):
    if(i%10==3 or i%10==7): #train-real
        shutil.copyfile(path_fake+fake_file[i], dest_test+'fake/'+fake_file[i])
    else: #val-real
        shutil.copyfile(path_fake+fake_file[i], dest_train+'fake/'+fake_file[i])
