import os
import re

dir_path = './satellite/images/test'
writepth = './images/test'
files = [f for f in os.listdir(dir_path) if f.endswith((".JPG",".jpg",".png",".tif"))]

outfile = open("test.txt", "w+")

for i, file in enumerate(files):
    outfile.write(writepth + '/' + file)
    outfile.write('\n')


dir_path = './satellite/images/train'
writepth = './images/train'
files = [f for f in os.listdir(dir_path) if f.endswith((".JPG",".jpg",".png",".tif"))]

outfile = open("train.txt", "w+")

for i, file in enumerate(files):
    outfile.write(writepth + '/' + file)
    outfile.write('\n')

dir_path = './satellite/images/val'
writepth = './images/val'

files = [f for f in os.listdir(dir_path) if f.endswith((".JPG",".jpg",".png",".tif"))]

outfile = open("val.txt", "w+")

for i, file in enumerate(files):
    outfile.write(writepth + '/' + file)
    outfile.write('\n')