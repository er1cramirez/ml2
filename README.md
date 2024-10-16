# ml2


## Instalaciones

### Conda(opcional)
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```
```
source miniconda3/bin/activate
conda init --all
```
```
conda create --name mi_entorno python=3.x
```
```
conda activate mi_entorno
```



### Torch
Conda
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
No conda
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
### Cuda


cuDNNN
```
wget https://developer.download.nvidia.com/compute/cudnn/9.5.0/local_installers/cudnn-local-repo-ubuntu2004-9.5.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2004-9.5.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2004-9.5.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn
```
```
sudo apt-get -y install cudnn-cuda-12
```

### Opencv Cuda
Install minimal prerequisites (Ubuntu 18.04 as reference)
```
sudo apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
```
```
sudo apt update && sudo apt install -y cmake g++ wget unzip
```
Download and unpack sources
```
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv.zip
unzip opencv_contrib.zip
```
Create build directory and switch into it
```
mkdir -p build && cd build
```
Configure
```
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x -D WITH_CUDA=ON -D CUDA_ARCH_BIN=7.5 -D CUDA_FAST_MATH=ON -D WITH_CUBLAS=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=OFF -D BUILD_EXAMPLES=OFF -D BUILD_opencv_python3=ON -D OPENCV_DNN_CUDA=ON -D OPENCV_ENABLE_NONFREE=ON -D OPENCV_DNN_CUDNN=ON -D PYTHON3_EXECUTABLE=$(which python3) -D PYTHON3_INCLUDE_DIR=$(python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])") -D PYTHON3_LIBRARY=$(python3 -c "from sysconfig import get_config_var; print(get_config_var('LIBDIR'))") -D PYTHON3_PACKAGES_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])") ..
```
Build
```
cmake --build .
```
