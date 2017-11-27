import os
import logging
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

gt_prefix = 'gt_'

flags.DEFINE_string('input_folder', None,
                    'Folder with input images.')


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
    input_folder = FLAGS.input_folder

    if input_folder is None:
        logging.error("No input_folder was given.")
        logging.info("Usage: python list_data.py --input_folder data/training")
        exit(1)

    if not os.path.exists(input_folder):
        logging.error("Input folder {} doesn't exist".format(FLAGS.input_folder))
        exit(1)

    input_dirs = os.listdir(input_folder)
    input_size = len(input_dirs)
    if input_size == 0:
        logging.error("Input folder {} is empty".format(FLAGS.input_folder))
        exit(1)
    elif input_size == 1:
        listTestData(os.path.join(input_folder, input_dirs[0]), os.path.join(input_folder, "test.txt"))
    elif input_size == 2:
        listTrainData(input_folder, input_dirs, os.path.join(input_folder, "train.txt"))
    else:
        logging.error("Input folder {} contains too many files/folders".format(FLAGS.input_folder))
        exit(1)


if __name__ == '__main__':
    main()
