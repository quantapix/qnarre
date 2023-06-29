wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update

sudo apt-get install --no-install-recommends \
    cuda-12-2 \
    libcudnn8=8.9.2.26-1+cuda12.1 \
    libcudnn8-dev=8.9.2.26-1+cuda12.1 \
    libnccl2=2.18.3-1+cuda12.1 \
    libnccl-dev=2.18.3-1+cuda12.1

sudo apt-get install -y --no-install-recommends \
    libnvinfer8=8.6.1.6-1+cuda12.0 \
    libnvinfer-dev=8.6.1.6-1+cuda12.0 \
    libnvinfer-plugin8=8.6.1.6-1+cuda12.0

function lib_installed() { /sbin/ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep $1; }
function check() { lib_installed $1 && echo "$1 is installed" || echo "ERROR: $1 is NOT installed"; }
check libcuda
check libcudart
check libcudnn
check libnccl
check libnvinfer
