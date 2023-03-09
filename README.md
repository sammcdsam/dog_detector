# Dog Detector
A dog detector built using YOLOV3  and Tensorflow in python. It looks for either dog in the frame and sends me a photo using the Discord API.  Designing to be able to snoop on my dogs, and building it as a Discord bot for a layer of security between my computer of livestreaming data and the internet. 

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

## About the files

### dog_detector.py
Uses OpenCV and YOLOv3 to save an image and a GIF when the dogs are detected in the frame. Logic added to not record people. 

### image.py
Runs YOLOv3 on a provided image. 

### utils.py
Various utilities used in dog_detector.py 
Non_max_suppression, resize_images, load_class_names, output_boxes (finds the bounding boxes), draw_ouputs (places bounding boxes on the opencv frame)

### yolov3.py
Read in the YOLO config file and create the network in tensorflow based on the config file for the network. 

### convert_weights.py
Load the model weights and modify them into the Tensorflow wieghts order. 

## Improvements
Train a model on just images of the dogs instead of using a pretrained YOLO model that has ~75 labels for objects.
Train a model that can detect which dog is in the frame
Implement a method to reduce the amount of time YOLO is running, maybe check for motion of some kind. 
Add some form of updated state  so a new picture is shared, but only once every certain amount of time
Maybe add a twitter account (but I dont want to pay for the API)


