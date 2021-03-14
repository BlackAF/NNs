#!/bin/bash

# Install Docker
echo "Installing docker"
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
  
# Install nvidia docker
echo "Installing nvidia docker"
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list   
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Install nvdia driver
echo "Installing nvidia driver"
sudo apt-get install --no-install-recommends nvidia-driver-450 -y

# Build the image
echo "Building docker image"
sudo docker build -t dl .

# Add aliases
echo "Adding aliases"
echo 'alias docker="sudo docker"' >> ~/.bashrc
echo 'alias train="docker run -d --rm --gpus all --mount type=bind,src=`pwd`,dst=/app --name training --entrypoint runipy dl"' >> ~/.bashrc
echo 'alias train_rm="docker stop training"' >> ~/.bashrc
echo 'alias train_logs="docker logs training --follow"' >> ~/.bashrc
echo 'alias tensorboard="docker run -d --rm --mount type=bind,src=`pwd`,dst=/app --name tensorboard -p 6006:6006 dl tensorboard --bind_all"' >> ~/.bashrc
echo 'alias tensorboard_rm="docker stop tensorboard"' >> ~/.bashrc
source ~/.bashrc

echo "Done!"

