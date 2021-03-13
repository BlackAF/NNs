#!/bin/bash

# Install Docker
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
  
# Install nvidia docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list   
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Install nvdia driver
sudo apt-get install --no-install-recommends nvidia-driver-450 -y

# Build the image
sudo docker build -t dl .

# Add aliases
echo 'alias train="sudo docker run -d --rm --gpus all --mount type=bind,src=`pwd`,dst=/app --name training --entrypoint runipy dl"' >> ~/.bashrc
echo 'alias train_logs="sudo docker logs training"' >> ~/.bashrc
echo 'alias tensorboard="sudo docker run -d --rm --mount type=bind,src=`pwd`,dst=/app --name tensorboard -p 6006:6006 dl tensorboard --bind_all --logdir"' >> ~/.bashrc
source ~/.bashrc

