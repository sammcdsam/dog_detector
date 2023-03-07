
# import the opencv library
import tensorflow as tf
import numpy as np
import pandas as pd
import os, glob, cv2
import matplotlib.pyplot as plt
import time
from yolov3 import YOLOv3Net
from utils import load_class_names, output_boxes, draw_outputs, resize_image

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model_size = (416, 416, 3)
num_classes = 80
class_name = 'model_data/coco.names'

max_output_size = 100
max_output_size_per_class= 20
iou_threshold = 0.5
confidence_threshold = 0.5
cfgfile = 'model_data/yolov3.cfg'
weightfile = 'weights/yolov3_weights.tf'

def main():
# define a video capture object
    vid = cv2.VideoCapture(0)
    model = YOLOv3Net(cfgfile,model_size,num_classes)
    model.load_weights(weightfile)
    class_names = load_class_names(class_name)
    win_name = 'Yolov3 detection'
    cv2.namedWindow(win_name)

    frame_size = (vid.get(cv2.CAP_PROP_FRAME_WIDTH),
                  vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while(True):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        if not ret:
            break

        resized_frame = tf.expand_dims(frame, 0)
        resized_frame = resize_image(resized_frame, (model_size[0],model_size[1]))
        pred = model.predict(resized_frame)
        boxes, scores, classes, nums = output_boxes( \
            pred, model_size,
            max_output_size=max_output_size,
            max_output_size_per_class=max_output_size_per_class,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold)
                
        img = draw_outputs(frame, boxes, scores, classes, nums, class_names)

        for i in range(nums[0]):
            if int(classes[0][i]) == 16:
                print('Dog')
                cv2.imshow(win_name, img)
        stop = time.time()

        # Display the resulting frame
        #cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()