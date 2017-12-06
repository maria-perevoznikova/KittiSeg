import os

gt_prefix = 'gt_'
# input_folder = "/home/maria/Documents/MLdata/train"
# input_folder = "/home/maria/Documents/MLdata/val"
input_folder = "/home/maria/Documents/MLdata/test"

def listTestData(data_dir, output):
    files = [os.path.join(data_dir, line) for line in os.listdir(data_dir)]
    with open(output, 'w') as f:
        f.write('\n'.join(files))


def listTrainData(input_folder, input_dirs, output):
    dir0 = input_dirs[0]
    dir1 = input_dirs[1]
    if dir0.startswith(gt_prefix):
        (gt_dir, img_dir) = (os.path.join(input_folder, dir0), os.path.join(input_folder, dir1))
    else:
        (img_dir, gt_dir) = (os.path.join(input_folder, dir0), os.path.join(input_folder, dir1))
    img_files = [os.path.join(img_dir, line) for line in os.listdir(gt_dir)]
    gt_files = [os.path.join(gt_dir, line) for line in os.listdir(gt_dir)]
    files = [im + ' ' + gt for im, gt in zip(img_files, gt_files)]
    with open(output, 'w') as f:
        f.write('\n'.join(files))


def main():
    if not os.path.exists(input_folder):
        print("Input folder {} doesn't exist".format(input_folder))
        exit(1)

    input_dirs = os.listdir(input_folder)
    input_size = len(input_dirs)
    if input_size == 0:
        print("Input folder {} is empty".format(input_folder))
        exit(1)
    elif input_size == 1:
        listTestData(os.path.join(input_folder, input_dirs[0]), os.path.join(input_folder, "test.txt"))
    elif input_size == 2:
        listTrainData(input_folder, input_dirs, os.path.join(input_folder, "train.txt"))
    else:
        print("Input folder {} contains too many files/folders".format(input_folder))
        exit(1)


if __name__ == '__main__':
    main()
