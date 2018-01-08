#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the model network using a queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import scipy as scp
import tensorvision.utils as utils


def eval_image(hypes, gt_image, output_image):
    """."""
    classes = hypes['classes']
    num_classes = len(classes)
    gt_labels = np.zeros((gt_image.shape[0], gt_image.shape[1], num_classes))
    # reshape gt_image: [H W n_chan] -> [H W n_cl]
    for i, k in enumerate(sorted(classes.keys())):
        color = classes[k]
        gt_labels[:,:,i] = np.all(gt_image == color, axis=2)

    # reshape gt_labels: [WxH n_cl]
    labels = np.reshape(gt_labels, (-1, num_classes))

    # TODO how to handle situation with 'unknown' class?
    # output_image [WxH n_cl], pred shape [WxH]
    pred = np.argmax(output_image, axis=1)

    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    tn = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    for i in range(num_classes):
        positive = np.equal(pred, i)
        tp[i] = np.sum(positive * labels[:, i])
        other_labels = np.sum(labels, 1) - labels[:, i]
        fp[i] = np.sum(positive * other_labels)

        negativ = np.not_equal(pred, i)
        fn[i] = np.sum(negativ * labels[:, i])
        tn[i] = np.sum(negativ * other_labels)

        # assertion holds only if each pixel belongs to one of the defined classes
        # i.e. no 'unknown' class is present
        # assert (labels.shape[0] == tp[i] + fp[i] + fn[i] + tn[i])

    return tp, fp, tn, fn


def evaluate(hypes, sess, image_pl, inf_out):

    softmax = inf_out['softmax']
    data_dir = hypes['dirs']['data_dir']
    num_classes = hypes['arch']['num_classes']

    # create colormap
    classes = hypes['classes']
    color_dict = {}
    for i, k in enumerate(sorted(classes.keys())):
        # add alpha channel
        color = list(classes[k])
        color.append(127)
        color_dict[i] = color
    print("Eval classes: {}".format(classes))
    print("Eval color dict: {}".format(color_dict))

    for phase in ['train', 'val']:
        data_file = hypes['data']['{}_file'.format(phase)]
        data_file = os.path.join(data_dir, data_file)
        image_dir = os.path.dirname(data_file)

        total_tp = np.zeros(num_classes)
        total_fp = np.zeros(num_classes)
        total_tn = np.zeros(num_classes)
        total_fn = np.zeros(num_classes)

        image_list = []
        with open(data_file) as file:
            for i, datum in enumerate(file):
                datum = datum.rstrip()
                image_file, gt_file = datum.split(" ")
                image_file = os.path.join(image_dir, image_file)
                gt_file = os.path.join(image_dir, gt_file)

                image = scp.misc.imread(image_file, mode='RGB')
                gt_image = scp.misc.imread(gt_file, mode='RGB')

                input_image, gt_image = _add_jitter(hypes, image, gt_image)

                shape = input_image.shape

                feed_dict = {image_pl: input_image}

                output = sess.run([softmax], feed_dict=feed_dict)
                output_im = output[0]
                output_im = _fix_shape_jitter(hypes, gt_image, output_im, shape)

                # gt_image shape [H W n_chan]
                # output_im shape [HxW n_cl]
                tp, fp, tn, fn = eval_image(hypes, gt_image, output_im)
                total_tp += tp
                total_fp += fp
                total_tn += tn
                total_fn += fn

                if phase == 'val':
                    _save_plot(image, output_im, color_dict, image_file, image_list)


    eval_list = []
    for phase in ['train', 'val']:
        for i in range(num_classes):
            tp = total_tp[i]
            fp = total_fp[i]
            tn = total_tn[i]
            fn = total_fn[i]
            total = tp + fp + tn + fn
            eval_list.append(('[{}] TP {}'.format(phase, i), tp / total))
            eval_list.append(('[{}] FP {}'.format(phase, i), fp / total))
            eval_list.append(('[{}] TN {}'.format(phase, i), tn / total))
            eval_list.append(('[{}] FN {}'.format(phase, i), fn / total))
            eval_list.append(('[{}] IoU {}'.format(phase, i), tp / (tp + fp + fn)))

        tp_ = np.sum(total_tp)
        fp_ = np.sum(total_fp)
        tn_ = np.sum(total_tn)
        fn_ = np.sum(total_fn)
        total = tp_ + fp_ + tn_ + fn_

        eval_list.append(('[{}] TP total '.format(phase), tp_ / total))
        eval_list.append(('[{}] FP total '.format(phase), fp_ / total))
        eval_list.append(('[{}] TN total '.format(phase), tn_ / total))
        eval_list.append(('[{}] FN total '.format(phase), fn_ / total))

        eval_list.append(('[{}] Acc. '.format(phase), (tn_ + tp_) / total))
        eval_list.append(('[{}] IoU '.format(phase), tp_ / (tp_ + fp_ + fn_)))

    return eval_list, image_list


def _save_plot(image, output_im, color_dict, image_file, image_list):
    # image shape [H W n_chan], output_im (softmax) shape [HxW n_cl]

    # keep values with probabilities > 0.5
    hard = output_im > 0.5
    hard_mask = np.sum(hard, axis=1) > 0
    segm_mask = np.argmax(output_im, axis=1) + 1
    # '-1' for unknown class (probability value < 0.5)
    segm_mask = segm_mask*hard_mask - 1
    segm_mask = np.reshape(segm_mask, (image.shape[0], image.shape[1]))

    # Saving overlay image
    ov_image = utils.overlay_segmentation(image, segm_mask, color_dict)
    name = os.path.basename(image_file)
    filename, file_extension = os.path.splitext(name)
    ov_name = filename + '_overlay' + file_extension
    image_list.append((ov_name, ov_image))

    # Saving segmentation image
    seg_image = utils.segmentation_rgb(segm_mask, color_dict)
    seg_name = filename + '_segm' + file_extension
    image_list.append((seg_name, seg_image))


def _fix_shape_jitter(hypes, gt_image, output_im, shape):
    if hypes['jitter']['fix_shape']:
        image_height = hypes['jitter']['image_height']
        image_width = hypes['jitter']['image_width']
        offset_x = (image_height - shape[0]) // 2
        offset_y = (image_width - shape[1]) // 2
        gt_shape = gt_image.shape
        output_im = output_im[offset_x:offset_x + gt_shape[0], offset_y:offset_y + gt_shape[1]]
    return output_im


def _add_jitter(hypes, image, gt_image):
    if hypes['jitter']['fix_shape']:
        shape = image.shape
        image_height = hypes['jitter']['image_height']
        image_width = hypes['jitter']['image_width']
        assert (image_height >= shape[0])
        assert (image_width >= shape[1])

        offset_x = (image_height - shape[0]) // 2
        offset_y = (image_width - shape[1]) // 2
        new_image = np.zeros([image_height, image_width, 3])
        new_image[offset_x:offset_x + shape[0], offset_y:offset_y + shape[1]] = image
        input_image = new_image
    elif hypes['jitter']['reseize_image']:
        image_height = hypes['jitter']['image_height']
        image_width = hypes['jitter']['image_width']
        input_image, gt_image = _resize_label_image(image, gt_image, image_height, image_width)
    else:
        input_image = image

    return input_image, gt_image

def _resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width), interp='cubic')
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width), interp='nearest')

    return image, gt_image