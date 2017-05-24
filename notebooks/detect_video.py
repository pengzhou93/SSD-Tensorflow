"""Detect video and save bbox to files
"""
import argparse
import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('../')

slim = tf.contrib.slim

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

def main(argv):
    # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    sess = tf.Session(config = config)
    # isess = tf.InteractiveSession(config=config)

    # Input placeholder.
    net_shape = (300, 300)
    data_format = 'NHWC'
    img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
    # Evaluation pre-processing: resize to SSD net shape.
    image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
        img_input, None, None, net_shape, data_format,
        resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    image_4d = tf.expand_dims(image_pre, 0)

    # Define the SSD model.
    reuse = True if 'ssd_net' in locals() else None
    ssd_net = ssd_vgg_300.SSDNet()
    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
        predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
    
    # Restore SSD model.
    ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
    # ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_filename)
    
    # SSD default anchor boxes.
    ssd_anchors = ssd_net.anchors(net_shape)    


    # Test on some demo image and visualize output.
    # import ipdb; ipdb.set_trace()
    if argv.output_dir is None:
        output_p_dir = os.path.dirname(argv.video_path)
        file_name = os.path.basename(argv.video_path).split('.')[0]
        output_dir = os.path.join(output_p_dir, file_name)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
    else:
        if not os.path.isdir(argv.output_dir):
            assert 0, "output_dir: {} is not exist.".format(argv.output_dir)
        output_dir = argv.output_dir
    cap = cv2.VideoCapture(argv.video_path)
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    count = 0
    name_template = 'frame-%06d.txt'
    while(cap.isOpened()):
        ret, img = cap.read()
        assert ret, "Close video"
        rclasses, rscores, rbboxes =  process_image(sess, img, image_4d,
                                                    predictions, localisations,
                                                    bbox_img, img_input, ssd_anchors)
        name = os.path.join(output_dir, name_template%count)
        visualization.save_bboxes_imgs_to_file(name, img, rclasses, rscores, rbboxes)    
        count += 1
        print("Detection frame [%d/%d]\r"%(count, num_frame), end = "")

        # assert 0
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break    

# Main image processing routine.
def process_image(sess, img, image_4d, predictions, localisations, bbox_img,
                  img_input, ssd_anchors, select_threshold=0.5,
                  nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = sess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

def parse_arguments():
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-vi", "--video_path", required=False, type = str,
                    default='/home/shhs/Downloads/cm/cm2.mp4', help="input video file")
    ap.add_argument("-o", "--output_dir", required=False, type = str,
                    default=None, help="output images dir")

    return ap.parse_args()

if __name__ == '__main__':
    argv = parse_arguments()
    main(argv)
