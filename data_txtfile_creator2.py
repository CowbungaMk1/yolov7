import os
import re

dir_path = './humveedata/Training_data/images/test'
writepth = './Training_data/images/test'
files = [f for f in os.listdir(dir_path) if f.endswith((".JPG",".jpg",".png",".tif"))]

outfile = open("test.txt", "w+")

for root, dirs, files in os.walk(dir_path, topdown=False):
    for dirk in dirs:
        for root3, dirs3, files3 in os.walk(os.path.join(root,dirk), topdown=False):
            for name in files3:

                print(os.path.join(writepth,dirk,name))
                outfile.write(os.path.join(writepth,dirk,name))
                outfile.write('\n')


dir_path = './humveedata/Training_data/images/train'
writepth = './Training_data/images/train'
outfile = open("train.txt", "w+")

for root, dirs, files in os.walk(dir_path, topdown=False):
    for dirk in dirs:
        for root3, dirs3, files3 in os.walk(os.path.join(root,dirk), topdown=False):
            for name in files3:

                print(os.path.join(writepth,dirk,name))
                outfile.write(os.path.join(writepth,dirk,name))
                outfile.write('\n')

dir_path = './humveedata/Training_data/images/val'
writepth = './Training_data/images/val'
outfile = open("val.txt", "w+")

for root, dirs, files in os.walk(dir_path, topdown=False):
    for dirk in dirs:
        for root3, dirs3, files3 in os.walk(os.path.join(root,dirk), topdown=False):
            for name in files3:

                print(os.path.join(writepth,dirk,name))
                outfile.write(os.path.join(writepth,dirk,name))
                outfile.write('\n')
