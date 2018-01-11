import os
import numpy as np
import scipy as scp
import scipy.misc


def main():
    input_folder = "C:\Projects\VP2\MLdata\ex2\\train\gt_tiles"
    # input_folder = "C:\Projects\VP2\MLdata\ex3\\train\gt_tiles"
    # input_folder = "C:\Projects\VP2\MLdata\\test"

    if not os.path.exists(input_folder):
        print("Input folder {} doesn't exist".format(input_folder))
        exit(1)

    input_files = os.listdir(input_folder)
    input_size = len(input_files)
    print("Input folder {} contains {} files".format(input_folder, input_size))

    colors = {
        "01": [255, 0, 0],
        "02": [0, 255, 0],
        "03": [0, 0, 255],
        "04": [255, 255, 0],
        "05": [255, 0, 255],
        "06": [0, 255, 255],
        "07": [100, 0, 0],
        "08": [0, 100, 0],
        "09": [0, 0, 100]
        # "10": [150, 150, 150]
    }

    # count pixels per class
    for file_name in input_files:
        count = np.zeros(len(colors), np.uint32)
        for i, k in enumerate(sorted(colors.keys())):
            gt_file = os.path.join(input_folder, file_name)
            gt_image = scp.misc.imread(gt_file, mode='RGB')

            color = colors[k]
            color_px = np.all(gt_image == color, axis=2)
            num_color_px = np.sum(color_px)
            count[i] += num_color_px
        print(count)

    num_px = gt_image.shape[0] * gt_image.shape[1]
    print(num_px, 'px total')

if __name__ == '__main__':
    main()
