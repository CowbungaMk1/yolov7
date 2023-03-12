import os
import re

dir_path = './Vermont_sim_data2/images/test'
writepth = './images/test'
files = [f for f in os.listdir(dir_path) if f.endswith(".jpg")]

outfile = open("test.txt", "w+")

for i, file in enumerate(files):
    outfile.write(writepth + '/' + file)
    outfile.write('\n')
dir_path = './Vermont_sim_data2/images/train'
writepth = './images/train'
files = [f for f in os.listdir(dir_path) if f.endswith(".jpg")]

outfile = open("train.txt", "w+")

for i, file in enumerate(files):
    outfile.write(writepth + '/' + file)
    outfile.write('\n')
    dir_path = './Vermont_sim_data2/images/val'
    writepth = './images/val'
    files = [f for f in os.listdir(dir_path) if f.endswith(".jpg")]

    outfile = open("val.txt", "w+")

    for i, file in enumerate(files):
        outfile.write(writepth + '/' + file)
        outfile.write('\n')