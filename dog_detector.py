# import the opencv library
#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import pandas as pd
import imageio
import os, glob, cv2
import matplotlib.pyplot as plt
import time
from yolov3 import YOLOv3Net
from utils import load_class_names, output_boxes, draw_outputs, resize_image, rescale_frame

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

    #load the model, weights and classes
    # TODO: Train a model on the difference in corgi color
    # TODO: Maybe upgrade to YOLOv5 or a different object detection network.
    model = YOLOv3Net(cfgfile,model_size,num_classes)
    model.load_weights(weightfile)
    class_names = load_class_names(class_name)

    dog_pic_saved = False
    dog_in_view = False
    person_in_view = False

    frame_size = (vid.get(cv2.CAP_PROP_FRAME_WIDTH),
                  vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_witdh = int(vid.get(3)/100)
    frame_height = int(vid.get(4)/100)

    size = (frame_witdh, frame_height)
    

    #store images to save a gif
    image_list = []
    frame_count = 0
    
    #
    while(True):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        # frame is being over written with the bounding box. I dont want to see the bounding box
        # so I am testing fixes right now
        backup_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        original_frame = frame

        if not ret:
            break

        dog_in_view = False
        person_in_view = False

        #predict the objects in the frame and add output boxes to the image using the classes
        resized_frame = tf.expand_dims(frame, 0)
        resized_frame = resize_image(resized_frame, (model_size[0],model_size[1]))
        pred = model.predict(resized_frame)
        boxes, scores, classes, nums = output_boxes( \
            pred, model_size,
            max_output_size=max_output_size,
            max_output_size_per_class=max_output_size_per_class,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold)
                
        #img = draw_outputs(frame, boxes, scores, classes, nums, class_names)

        # instead of displaying the image on the screen save an image of the dog
        # TODO: do NOT save the image when a person is detected in the frame
        # TODO: only save images that have both dogs detected
        for i in range(nums[0]):
            if int(classes[0][i]) == 16:
                dog_in_view = True
            if int(classes[0][i]) == 0:
                person_in_view = True
            
        if (dog_in_view and not person_in_view): 
            frame50 = rescale_frame(backup_frame, percent=50)
            image_list.append(frame50)
            frame_count += 1
        
        #save the first image of the frame as an image to be shared. 
        if frame_count == 1:
            cv2.imwrite("data/output_images/saved_dog_pic.jpg", original_frame)
        
        #create a 3 second gif of the following data
        if frame_count == 60:
            imageio.mimsave("data/output_images/saved_dog.gif", image_list, fps=20)
            image_list=[]
            frame_count=0

        
        #cv2.imshow(win_name, img)
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

    print("Video Saved")

if __name__ == '__main__':
    main()