import os
import numpy as np
import scipy as scp
import scipy.misc

def main():
    # input_folder = "C:\Projects\VP2\MLdata\\ex1\\train\gt_tiles"
    # output_folder = "C:\Projects\VP2\MLdata\\ex1\\train\gt_tiles_"
    input_folder = "C:\Projects\VP2\MLdata\\ex1\\val\gt_tiles"
    output_folder = "C:\Projects\VP2\MLdata\\ex1\\val\gt_tiles_"
    # input_folder = "C:\Projects\VP2\MLdata\\test"
    # output_folder = "C:\Projects\VP2\MLdata\\test_"

    if not os.path.exists(input_folder):
        print("Input folder {} doesn't exist".format(input_folder))
        exit(1)

    input_files = os.listdir(input_folder)
    input_size = len(input_files)
    print("Input folder {} contains {} files".format(input_folder, input_size))

    # replace white px with black
    count = 0
    black = np.zeros(3, dtype=np.uint8)
    white = 255 * np.ones(3, dtype=np.uint8)
    for file_name in input_files:
        gt_file = os.path.join(input_folder, file_name)
        gt_image = scp.misc.imread(gt_file, mode='RGB')

        width, height = gt_image.shape[0], gt_image.shape[1]
        for x in range(0, width):
            for y in range(0, height):
                if np.all(gt_image[x, y] == white):
                    gt_image.putpixel((y, x), black)

        gt_file_copy = os.path.join(output_folder, file_name)
        scp.misc.imsave(gt_file_copy, gt_image)

        count += 1

    print("{} gt files saved".format(count))

if __name__ == '__main__':
    main()
