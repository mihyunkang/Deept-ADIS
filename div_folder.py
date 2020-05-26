import os
import shutil

path_A = "../../data/deepfake/fb_dfd_release_0.1_final/method_A/"
path_B = "../../data/deepfake/fb_dfd_release_0.1_final/method_B/"
path_original = "../../data/deepfake/fb_dfd_release_0.1_final/original_videos/"

file_list_A = os.listdir(path_A)
file_list_B = os.listdir(path_B)
file_list_original = os.listdir(path_original)

dest_dir_A = "./data/fake_methodA/"
dest_dir_B = "./data/fake_methodB/"
dest_dir = "./data/real/"

for i in file_list_A:
    path = path_A+i+'/'
    file_list = os.listdir(path)
    for j in file_list:
        path1 = path + j + '/'
        fl = os.listdir(path1)
        for k in fl:
            path2 = path1 + k
            shutil.copy(path2, dest_dir_A)

for i in file_list_B:
    path = path_B+i+'/'
    file_list = os.listdir(path)
    for j in file_list:
        path1 = path + j + '/'
        fl = os.listdir(path1)
        for k in fl:
            path2 = path1 + k
            shutil.copy(path2, dest_dir_B)

for i in file_list_original:
    path = path_original+i+'/'
    file_list = os.listdir(path)
    for j in file_list:
        path1 = path + j
        shutil.copy(path1, dest_dir)

#kaggle deepfake dataset 