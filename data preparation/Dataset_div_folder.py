#train:test:val 폴더에 7:2:1 비율로 옮겨놓기.
#fake/real 데이터 각각!!
import os
import shutil

##### 파일 불러오기
#real 데이터
path_real = "D:/deeper_forensics/real"
#fake 데이터
path_fake = "D:/deeper_forensics/fake"
#파일경로로 불러옴
real_file = os.listdir(path_real)
fake_file = os.listdir(path_fake)

##### 옮기는 경로~~~
#test 데이터
dest_test = "D:/deeper_forensics/test/"
#train 데이터
dest_train = "D:/deeper_forensics/train/"
#val 데이터
dest_val = "D:/deeper_forensics/val/"

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
        shutil.move(path_fake+'/'+fake_file[i], dest_test+'fake'+fake_file[i])
    else: #val-real
        shutil.move(path_fake+'/'+fake_file[i], dest_val+'fake'+fake_file[i])
