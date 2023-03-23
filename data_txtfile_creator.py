import os
import re

dir_path = './humveedata/images/val'
writepth = './images/val'
files = [f for f in os.listdir(dir_path) if f.endswith(".jpg")]

outfile = open("val.txt", "w+")

for i, file in enumerate(files):
    outfile.write(writepth + '/' + file)
    outfile.write('\n')