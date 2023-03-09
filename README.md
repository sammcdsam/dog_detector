# Dog Detector
A dog detector built using YOLOV3  and Tensorflow in python. It looks for either dog in the frame and sends me a photo using the Discord API.  Designing to be able to snoop on my dogs. 

<p align="center">
  <img src="/data/output_images/dog_livestream.JPG" />
</p>

<p align="center">
  <img src="/data/output_images/saved_dog.gif" alt="animated" />
</p>

## Does not record people
Built to not record or save images that have a dog and a person in them. The GIF below is an example of this feature. I was taking Maggie on her walk and I am only record when walk out of the frame and close the door.

<p align="center">
  <img src="/data/output_images/no_person.gif" alt="animated" />
</p>

### dog_detector.py
Uses OpenCV and YOLOv3 to open a livestreaming camera feed that labels dogs in the frame

### image.py
Runs YOLOv3 on a provided image. 

### utils.py
Non_max_suppression, resize_images, load_class_names, output_boxes (finds the bounding boxes), draw_ouputs (places bounding boxes on the opencv frame)

### yolov3.py
Read in the YOLO config file and create the network in tensorflow based on the config file for the network. 

### convert_weights.py
Load the model weights and modify them into the Tensorflow wieghts order. 

### Improvements
Train a model on just images of the dogs instead of using a pretrained YOLO model that has ~75 labels for objects.
Train a model that can detect which dog is in the frame