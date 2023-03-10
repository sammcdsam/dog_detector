# A dockerfile must always start by importing the base image.
# We use the keyword 'FROM' to do that.
# In our example, we want import the python image.
# So we write 'python' for the image name and 'latest' for the version.
#FROM dustynv/jetson-inference:r32.7.1
 
FROM ubuntu:22.04
FROM python:latest
FROM tensorflow/tensorflow:devel-gpu

#FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/targets/x86_64-linux/lib

#COPY /usr/local/cuda-12.1/targets/x86_64-linux/lib/ /user/local/cuda-12.1/targets/x86_64-linux/lib/
#RUN ldconfig

#ENV cudnn_version=8.8.0.121
#ENV cuda_version=cuda12.1

#ENV NVIDIA_VISIBLE_DEVICES all
#ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
#ENV NVIDIA_REQUIRE_CUDA "cuda>=12.0"

#install python and some other libraries
RUN apt-get update && yes | apt-get upgrade
RUN apt install libcublas-12-1
RUN apt-get install -y python3-pip
RUN apt-get update && apt-get install -y libglib2.0-0 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
#RUN apt install nvidia-container-toolkit

# download requirements - need to update requirements.txt, but have to confirm versions
RUN pip install tensorflow
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
#CMD nvidia-smi
#CMD ["nvcc --version"]
CMD ["./run.sh"]