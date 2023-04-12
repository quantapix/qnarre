wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update

sudo apt-get install --no-install-recommends \
    cuda-12-1 \
    libcudnn8=8.9.0.131-1+cuda12.1 \
    libcudnn8-dev=8.9.0.131-1+cuda12.1 \
    libnccl2=2.17.1-1+cuda12.1 \
    libnccl-dev=2.17.1-1+cuda12.1

sudo apt-get install -y --no-install-recommends \
    libnvinfer8=8.6.0.12-1+cuda12.0 \
    libnvinfer-dev=8.6.0.12-1+cuda12.0 \
    libnvinfer-plugin8=8.6.0.12-1+cuda12.0
