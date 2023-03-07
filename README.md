# Dog Detector
A dog detector built using YOLOV3  and Tensorflow in python. It looks for 2 dogs in the frame and sends me a photo using the Discord API.  Designing to be able to snoop on my dogs. 

<p align="center">
  <img src="/data/images/dog_livestream.JPG" />
</p>


# dog_detector.py
Uses OpenCV and YOLOv3 to open a livestreaming camera feed that labels dogs in the frame

# image.py
Runs YOLOv3 on a provided image. 

# utils.py
Non_max_suppression, resize_images, load_class_names, output_boxes (finds the bounding boxes), draw_ouputs (places bounding boxes on the opencv frame)

# yolov3.py
Read in the YOLO config file and create the network in tensorflow based on the config file for the network. 

# convert_weights.py
Load the model weights and modify them into the Tensorflow wieghts order. 

# Improvements
Train a model on just images of the dogs instead of using a pretrained YOLO model that has ~75 labels for objects.