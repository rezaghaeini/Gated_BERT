#!/bin/bash

# In case of trouble in runing file:
# chmod 775 setup_routin.sh

# To install Nvidia-Driver (especially for K80 servers):
# sudo apt install nvidia-340
# sudo apt install nvidia-utils-390
# sudo apt install nvidia-driver-390

mkdir Downloads
mkdir projects
cd Downloads
# COMMENT: downloading CUDA 10.0
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
# COMMENT: installing gcc and g++
sudo apt install gcc
# COMMENT: installing ipython
sudo apt install python3-pip
# COMMENT" installing CUDA 10.0
sudo sh cuda_10.0.130_410.48_linux

# TODO / INFO
# How to install CUDA 10.0:
#
# Do you accept the previously read EULA?
# accept/decline/quit: accept
#
# Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 410.48?
# (y)es/(n)o/(q)uit: y
#
# Do you want to install the OpenGL libraries?
# (y)es/(n)o/(q)uit [ default is yes ]: y
#
# Do you want to run nvidia-xconfig?
# This will update the system X configuration file so that the NVIDIA X driver
# is used. The pre-existing X configuration file will be backed up.
# This option should not be used on systems that require a custom
# X configuration, such as systems with multiple GPU vendors.
# (y)es/(n)o/(q)uit [ default is no ]: n
#
# Install the CUDA 10.0 Toolkit?
# (y)es/(n)o/(q)uit: y
#
# Enter Toolkit Location
#  [ default is /usr/local/cuda-10.0 ]:
#
# Do you want to install a symbolic link at /usr/local/cuda?
# (y)es/(n)o/(q)uit: y
#
# Install the CUDA 10.0 Samples?
# (y)es/(n)o/(q)uit: n

nano ~/.bashrc

# TODO
# Copy and Past the following line in your .bashrc file and save it:
#
# export PATH=$PATH:/usr/local/cuda-10.0/bin
# export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
# export CUDA_ROOT=/usr/local/cuda-10.0/bin
# alias pip=pip3
# alias python=python3
# alias wgpu="watch -n 1 nvidia-smi"
# alias nsmi=nvidia-smi

# COMMENT: entabling persistence mode for GPUs
sudo nvidia-smi -pm 1
sudo apt install unzip
# COMMENT: installing virtualenv
sudo apt-get install python-virtualenv
sudo apt-get install python-setuptools


# TODO
# Run the following commands next:
#cd ..
#mkdir vs
#cd vs
#virtualenv venv

# cd ..
# cd projects
# git clone https://fuselabs.visualstudio.com/APUB/_git/APUB
# cd APUB/interns/reza/itp-mt-dnn