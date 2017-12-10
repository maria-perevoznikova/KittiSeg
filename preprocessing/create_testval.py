import os
import math
import random


def main():
    gt_folder = "C:\Projects\VP2\MLdata\gt_tiles"
    orthos_folder = "C:\Projects\VP2\MLdata\orthos_tiles"

    val_folder_gt_tiles = "C:\Projects\VP2\MLdata\\val\gt_tiles"
    val_folder_tiles = "C:\Projects\VP2\MLdata\\val\\tiles"

    train_folder_gt_tiles = "C:\Projects\VP2\MLdata\\train\gt_tiles"
    train_folder_tiles = "C:\Projects\VP2\MLdata\\train\\tiles"

    if not os.path.exists(gt_folder):
        print("Input folder {} doesn't exist".format(gt_folder))
        exit(1)

    input_files = os.listdir(gt_folder)
    input_size = len(input_files)
    print("Input folder {} contains {} files".format(gt_folder, input_size))

    # validation set (20% of available data)
    val_size = int(input_size * 0.2)
    print("Validation set size: {}".format(val_size))
    random.shuffle(input_files)
    print("Shuffled files: {}".format(input_files[:10]))

    # move first 'val_size' files to val folder
    for i in range(val_size):
        file_name = input_files[i]
        os.rename(os.path.join(gt_folder, file_name), os.path.join(val_folder_gt_tiles, file_name))
        os.rename(os.path.join(orthos_folder, file_name), os.path.join(val_folder_tiles, file_name))

    # move the rest to train folder
    for i in range(val_size, input_size):
        file_name = input_files[i]
        os.rename(os.path.join(gt_folder, file_name), os.path.join(train_folder_gt_tiles, file_name))
        os.rename(os.path.join(orthos_folder, file_name), os.path.join(train_folder_tiles, file_name))


if __name__ == '__main__':
    main()
