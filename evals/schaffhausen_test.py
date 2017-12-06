#!/usr/bin/env python
# pylint: disable=missing-docstring
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the model network using a queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os.path
import sys

import scipy as scp
import scipy.misc

sys.path.insert(1, '../../submodules')

import tensorflow as tf

import tensorvision.utils as utils
import tensorvision.core as core

from seg_utils import seg_utils as seg

# configure logging
if 'TV_IS_DEV' in os.environ and os.environ['TV_IS_DEV']:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

flags = tf.app.flags
FLAGS = flags.FLAGS


def create_test_output(hypes, sess, image_pl, softmax):
    data_file = hypes['data']['test_file']
    image_dir = os.path.dirname(data_file)

    logdir = "test_images/"
    logdir_rb = "test_images_rb/"
    logdir_green = "test_images_green/"

    logging.info("Images will be written to {}/test_images_{{green, rg}}"
                 .format(logdir))

    logdir = os.path.join(hypes['dirs']['output_dir'], logdir)
    logdir_rb = os.path.join(hypes['dirs']['output_dir'], logdir_rb)
    logdir_green = os.path.join(hypes['dirs']['output_dir'], logdir_green)

    num_classes = hypes['arch']['num_classes']

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    if not os.path.exists(logdir_rb):
        os.mkdir(logdir_rb)

    if not os.path.exists(logdir_green):
        os.mkdir(logdir_green)

    with open(data_file) as file:
        for i, image_file in enumerate(file):
                image_file = image_file.rstrip()
                image_file = os.path.join(image_dir, image_file)
                image = scp.misc.imread(image_file)
                shape = image.shape

                feed_dict = {image_pl: image}

                output = sess.run([softmax['softmax']], feed_dict=feed_dict)

                output_images = []
                ov_images = []
                green_images = []
                for i in range(num_classes):
                    output_im = output[0][:, i].reshape(shape[0], shape[1])
                    output_images.append(output_im)
                    ov_images.append(seg.make_overlay(image, output_im))
                    hard = output_im > 0.5
                    green_images.append(utils.fast_overlay(image, hard))

                full_name = os.path.basename(image_file)
                name = os.path.splitext(full_name)[0]
                ext = os.path.splitext(full_name)[1]
                logging.info("Writing file: %s", full_name)
                for i in range(num_classes):
                    name_i = name + str(i) + '.' + ext

                    save_file = os.path.join(logdir, name_i)
                    scp.misc.imsave(save_file, output_images[i])

                    save_file = os.path.join(logdir_rb, name_i)
                    scp.misc.imsave(save_file, ov_images[i])

                    save_file = os.path.join(logdir_green, name_i)
                    scp.misc.imsave(save_file, green_images[i])


def _create_input_placeholder():
    image_pl = tf.placeholder(tf.float32)
    label_pl = tf.placeholder(tf.float32)
    return image_pl, label_pl


def infer(logdir):
    """
    Analyze a trained model.

    This will load model files and weights found in logdir and run a basic
    analysis.

    Parameters
    ----------
    logdir : string
        Directory with logs.
    """
    hypes = utils.load_hypes_from_logdir(logdir)
    modules = utils.load_modules_from_logdir(logdir)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():

        # prepaire the tv session

        with tf.name_scope('Validation'):
            image_pl, label_pl = _create_input_placeholder()
            image = tf.expand_dims(image_pl, 0)
            softmax = core.build_inference_graph(hypes, modules,
                                                 image=image)

        sess = tf.Session()
        saver = tf.train.Saver()

        core.load_weights(logdir, sess, saver)

        create_test_output(hypes, sess, image_pl, softmax)
    return


def main(_):
    """Run main function."""
    if FLAGS.logdir is None:
        logging.error("No logdir are given.")
        logging.error("Usage: tv-analyze --logdir dir")
        exit(1)

    if FLAGS.gpus is None:
        if 'TV_USE_GPUS' in os.environ:
            if os.environ['TV_USE_GPUS'] == 'force':
                logging.error('Please specify a GPU.')
                logging.error('Usage tv-train --gpus <ids>')
                exit(1)
            else:
                gpus = os.environ['TV_USE_GPUS']
                logging.info("GPUs are set to: %s", gpus)
                os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    else:
        logging.info("GPUs are set to: %s", FLAGS.gpus)
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus

    utils.load_plugins()

    logdir = os.path.realpath(FLAGS.logdir)

    logging.info("Starting to analyze Model in: %s", logdir)
    infer(logdir)


if __name__ == '__main__':
    tf.app.run()
