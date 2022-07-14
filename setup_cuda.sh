wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update

wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
sudo apt-get update

# wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
# sudo apt install ./libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
# sudo apt-get update

sudo apt-get install --no-install-recommends \
    cuda-11-5 \
    libcudnn8=8.3.2.44-1+cuda11.5  \
    libcudnn8-dev=8.3.2.44-1+cuda11.5 \
    libnccl2=2.11.4-1+cuda11.5 \
    libnccl-dev=2.11.4-1+cuda11.5

# sudo apt-get install -y --no-install-recommends \
#    libnvinfer7=7.1.3-1+cuda11.0 \
#    libnvinfer-dev=7.1.3-1+cuda11.0 \
#    libnvinfer-plugin7=7.1.3-1+cuda11.0
