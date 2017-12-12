import os
import numpy as np
import scipy as scp
import scipy.misc

def main():
    input_folder = "C:\Projects\VP2\MLdata\gt_tiles"
    # input_folder = "C:\Projects\VP2\MLdata\\test"
    dest_folder = "C:\Projects\VP2\MLdata\\black_tiles"

    if not os.path.exists(input_folder):
        print("Input folder {} doesn't exist".format(input_folder))
        exit(1)

    input_files = os.listdir(input_folder)
    input_size = len(input_files)
    print("Input folder {} contains {} files".format(input_folder, input_size))

    # move image with more than 95% black pixels to other folder
    count = 0
    black = np.zeros(3, dtype=np.uint8)
    for file_name in input_files:
        gt_file = os.path.join(input_folder, file_name)
        gt_image = scp.misc.imread(gt_file, mode='RGB')

        black_px = np.all(gt_image == black, axis=2)
        num_black_px = np.sum(black_px)
        num_px = gt_image.shape[0] * gt_image.shape[1]
        if num_black_px / num_px > 0.95:
            os.rename(gt_file, os.path.join(dest_folder, file_name))
            count += 1

    print("{} black files found".format(count))

if __name__ == '__main__':
    main()
