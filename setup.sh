# wget -P external https://github.com/microsoft/onnxruntime/releases/download/v1.21.1/onnxruntime-linux-x64-gpu-1.21.1.tgz
# cd external
# tar -xvzf onnxruntime-linux-x64-gpu-1.21.1.tgz

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cudnn-cuda-12