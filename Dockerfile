# A dockerfile must always start by importing the base image.
# We use the keyword 'FROM' to do that.
# In our example, we want import the python image.
# So we write 'python' for the image name and 'latest' for the version.
#FROM dustynv/jetson-inference:r32.7.1
FROM nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04
FROM python:latest
FROM tensorflow/tensorflow:latest-gpu

#ENV cudnn_version=8.4.0.26
#ENV cuda_version=cuda11.6

#install python and some other libraries
RUN apt-get update && yes | apt-get upgrade
RUN apt-get install -y python3-pip
RUN apt-get update && apt-get install -y libglib2.0-0 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
#RUN apt install nvidia-container-toolkit

# download requirements - need to update requirements.txt, but have to confirm versions
RUN pip install tensorflow==2.8.0
RUN pip install discord.py
RUN pip install pandas
RUN pip install imageio
RUN pip install opencv-python
RUN pip install matplotlib
RUN pip install python-dotenv

#RUN mkdir /usr/include

# Copy the file structure
#COPY /usr/include/cudnn.h /usr/include
COPY ./ ./

# Run both the python files in the script
RUN chmod a+x run.sh
#RUN pip install -r requirements.txt
#CMD ["nvidia-smi"]
#CMD ["nvcc --version"]
CMD ["./run.sh"]