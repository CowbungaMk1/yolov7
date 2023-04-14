import os
import re

dir_path = './satellite/images/test'
writepth = './images/test'
files = [f for f in os.listdir(dir_path) if f.endswith((".JPG",".jpg",".png",".tif"))]

outfile = open("test.txt", "w+")

for root, dirs, files in os.walk(dir_path, topdown=False):
    for dirk in dirs:
        for root3, dirs3, files3 in os.walk(os.path.join(root,dirk), topdown=False):
            for name in files3:

                print(os.path.join(writepth,dirk,name))
                outfile.write(os.path.join(writepth,dirk,name))
                outfile.write('\n')


dir_path = './satellite/images/train'
writepth = './images/train'
outfile = open("train.txt", "w+")

for root, dirs, files in os.walk(dir_path, topdown=False):
    for dirk in dirs:
        for root3, dirs3, files3 in os.walk(os.path.join(root,dirk), topdown=False):
            for name in files3:

                print(os.path.join(writepth,dirk,name))
                outfile.write(os.path.join(writepth,dirk,name))
                outfile.write('\n')

dir_path = './satellite/images/val'
writepth = './images/val'
outfile = open("val.txt", "w+")

for root, dirs, files in os.walk(dir_path, topdown=False):
    for dirk in dirs:
        for root3, dirs3, files3 in os.walk(os.path.join(root,dirk), topdown=False):
            for name in files3:

                print(os.path.join(writepth,dirk,name))
                outfile.write(os.path.join(writepth,dirk,name))
                outfile.write('\n')
