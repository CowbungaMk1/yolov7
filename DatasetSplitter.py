import pandas as pd

import argparse
import os
import shutil
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='test program')
parser.add_argument('-init', '--settings', help='print the supplied argument.', nargs='*')
args = parser.parse_args()
"""




"""

def main():
    if len(args.settings) < 3:
        print('too few arguments')
        print('example: python3 DatasetSplitter.py -init datasets/sodascapes/SODA_Labels datasets/sodascapes/InfraredImagesPNG datasets/sodascapes/')
        return 0
    else:
        print(
            'example: python3 DatasetSplitter.py -init datasets/sodascapes/SODA_Labels datasets/sodascapes/InfraredImagesPNG datasets/sodascapes/')
        labels_dir = args.settings[0]
        frames_dir = args.settings[1]
        results_dir = args.settings[2]

        print(frames_dir, labels_dir, results_dir)

    # sorting the frames and labels and saving list of names and directories



    frames = sorted(os.listdir(frames_dir))
    labels = sorted(os.listdir(labels_dir))

    # Let's say we want to split the data in 80:10:10 for train:valid:test dataset

    # In the first step we will split the data in training and remaining dataset
    X_train, X_rem, y_train, y_rem = train_test_split(frames, labels, train_size=0.7)

    # Now since we want the valid and test size to be equal (10% each of overall data).
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    test_size = 0.5
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

    print(len(X_train)), print(len(y_train))
    print(len(X_test)), print(len(y_test))
    print(len(X_valid)), print(len(y_valid))
    # print(X_train)

    # Saving the splits intro their respective directors
    sourcefolder = frames_dir.split('/')[len(frames_dir.split('/'))-1]
    print(sourcefolder)

    for i in range(len(X_train)):

        folder_name = [os.path.join(labels_dir, y_train[i]), os.path.join(results_dir, 'labels', 'train',sourcefolder)]
        os.makedirs(folder_name[1], exist_ok=True) 
        shutil.copy(folder_name[0],folder_name[1])

        folder_name = [os.path.join(frames_dir, X_train[i]), os.path.join(results_dir, 'images', 'train',sourcefolder)]
        os.makedirs(folder_name[1], exist_ok=True) 

        shutil.copy(folder_name[0],folder_name[1])


    for i in range(len(X_test)):

        folder_name = [os.path.join(labels_dir, y_test[i]), os.path.join(results_dir, 'labels', 'test',sourcefolder)]
        os.makedirs(folder_name[1], exist_ok=True) 

        shutil.copy(folder_name[0],folder_name[1])

        folder_name = [os.path.join(frames_dir, X_test[i]), os.path.join(results_dir, 'images', 'test',sourcefolder)]
        os.makedirs(folder_name[1], exist_ok=True)
        shutil.copy(folder_name[0],folder_name[1])


    for i in range(len(X_valid)):

        folder_name = [os.path.join(labels_dir, y_valid[i]), os.path.join(results_dir, 'val', 'labels',sourcefolder)]
        os.makedirs(folder_name[1], exist_ok=True)
        shutil.copy(folder_name[0],folder_name[1])


        folder_name = [os.path.join(frames_dir, X_valid[i]), os.path.join(results_dir, 'images', 'val',sourcefolder)]
        os.makedirs(folder_name[1], exist_ok=True)
        shutil.copy(folder_name[0],folder_name[1])


if __name__ == '__main__':
    main()
