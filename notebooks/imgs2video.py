#!/usr/bin/env python
import cv2
import argparse
import os
import sys

def main(argv):
    # Arguments
    dir_path = argv['imgs_path']
    ext = argv['extension']
    output = argv['output']
    
    images = []
    for f in os.listdir(dir_path):
        if f.endswith(ext):
            images.append(f)
    assert len(images), "Don't have images in dir : {}".format(dir_path)
        
    # Determine the width and height from the first image
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    cv2.imshow('video',frame)
    height, width, channels = frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))
    
    for image in images:
    
        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)
    
        out.write(frame) # Write out frame to video
    
        cv2.imshow('video',frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break
    
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

    print("The output video is {}".format(output))

def parse_arguments(argv):
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imgs_path", required=False, type = str,
                    default='.', help="input images file")
    ap.add_argument("-ext", "--extension", required=False, type = str,
                    default='png', help="extension name. default is 'png'.")
    ap.add_argument("-o", "--output", required=False, type = str,
                    default='output.mp4', help="output video file")

    args = vars(ap.parse_args(argv))
    return args
    
    
if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)

