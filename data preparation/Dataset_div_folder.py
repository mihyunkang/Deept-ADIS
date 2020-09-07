#train:test:val 폴더에 7:2:1 비율로 옮겨놓기.
#fake/real 데이터 각각!!
import os
import shutil

##### 파일 불러오기
#real 데이터
path_real = "D:/data/deeper_forensics/real/"
#fake 데이터
<<<<<<< HEAD
path_fake = "D:/data/deeper_forensics/fake/"
=======
path_fake = "D:/deeper_forensics/fake" #path_fake = "D:/full_data/fake"
>>>>>>> 836384cfb2f62c382b9140ec21001d70541ee65e
#파일경로로 불러옴
real_file = os.listdir(path_real)
fake_file = os.listdir(path_fake)

##### 옮기는 경로~~~
#test 데이터
dest_test = "D:/data/deeper_forensics/test/"
#train 데이터
dest_train = "D:/data/deeper_forensics/train/"
#val 데이터
dest_val = "D:/data/deeper_forensics/val/"

#real file - 7:2:1 분할
for i in range(len(real_file)):
    if(i<len(real_file)*0.7): #train-real
        shutil.move(path_real+'/'+real_file[i], dest_train+'real/'+real_file[i])
    elif(len(real_file)*0.7<i and i <=len(real_file)*0.9): #test-real
        shutil.move(path_real+'/'+real_file[i], dest_test+'real/'+real_file[i])
    else: #val-real
        shutil.move(path_real+'/'+real_file[i], dest_val+'real/'+real_file[i])

#fake file - 7:2:1 분할
for i in range(len(fake_file)):
    if(i<len(fake_file)*0.7): #train-real
        shutil.move(path_fake+'/'+fake_file[i], dest_train+'fake/'+fake_file[i])
    elif(len(fake_file)*0.7<i and i <=len(fake_file)*0.9): #test-real
        shutil.move(path_fake+'/'+fake_file[i], dest_test+'fake/'+fake_file[i])
    else: #val-real
        shutil.move(path_fake+'/'+fake_file[i], dest_val+'fake/'+fake_file[i])
