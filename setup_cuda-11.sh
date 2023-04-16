wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update

sudo apt-get install --no-install-recommends \
    cuda-11-8 \
    libcudnn8-dev=8.8.1.3-1+cuda11.8 \
    libcudnn8=8.8.1.3-1+cuda11.8 \
    libnccl2=2.15.5-1+cuda11.8 \
    libnccl-dev=2.15.5-1+cuda11.8

sudo apt-get install -y --no-install-recommends \
    libnvinfer8=8.6.0.12-1+cuda11.8 \
    libnvinfer-dev=8.6.0.12-1+cuda11.8 \
    libnvinfer-plugin8_8.6.0.12-1+cuda11.8

function lib_installed() { /sbin/ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep $1; }
function check() { lib_installed $1 && echo "$1 is installed" || echo "ERROR: $1 is NOT installed"; }
check libcuda
check libcudart
check libcudnn
check libnccl
check libnvinfer
