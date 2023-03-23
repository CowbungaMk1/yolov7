import os
import re

dir_path = './humveedata/images/test'
writepth = './images/test'
files = [f for f in os.listdir(dir_path) if f.endswith((".JPG",".jpg",".png",".tif"))]

outfile = open("test.txt", "w+")

for i, file in enumerate(files):
    outfile.write(writepth + '/' + file)
    outfile.write('\n')