import os


def main():
    gt_tiles_folder = "C:\Projects\VP2\MLdata\gt_tiles"
    # train_folder_old = "C:\Projects\VP2\MLdata\\ex2\\train\gt_tiles"
    # train_folder_new = "C:\Projects\VP2\MLdata\\ex1\\train\gt_tiles"
    train_folder_old = "C:\Projects\VP2\MLdata\\ex2\\val\gt_tiles"
    train_folder_new = "C:\Projects\VP2\MLdata\\ex1\\val\gt_tiles"

    if not os.path.exists(train_folder_old):
        print("Input folder {} doesn't exist".format(train_folder_old))
        exit(1)

    input_files = os.listdir(train_folder_old)
    input_size = len(input_files)
    print("Input folder {} contains {} files".format(train_folder_old, input_size))

    count = 0
    for file_name in input_files:
        gt_file = os.path.join(gt_tiles_folder, file_name)
        os.rename(gt_file, os.path.join(train_folder_new, file_name))
        count += 1

    print("{} gt files moved".format(count))

if __name__ == '__main__':
    main()
