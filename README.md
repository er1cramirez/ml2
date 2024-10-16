
# Cuda Install

Get started
First, verify that you have CUDA-capable NVIDIA hardware. To do this, run the following command and expect an output similar to this one:
```
lspci | grep -i nvidia
```
```
sudo apt install linux-headers-$(uname -r)
```
```
sudo apt-get update
```
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2004-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get updatesudo apt-get -y install cuda-toolkit-12-4
```
```
sudo apt-get install -y nvidia-driver-550-open
sudo apt-get install -y cuda-drivers-550
```
```
sudo reboot
```
```
code ~/.bashrc
```
```
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
Test the installation
```
nvcc — version
```


# Bebop Driver
```
sudo apt-get update
```
```
cd ~/
mkdir -p bebop_ws/src && cd ~/bebop_ws/src
git clone https://github.com/antonellabarisic/parrot_arsdk.git
cd parrot_arsdk
git checkout noetic_dev
```
```
sudo apt-get install libavahi-client-dev
```
```
sudo ln -s /usr/bin/python3 /usr/bin/python
cd ~/bebop_ws
catkin_make
```
```
cd src
git clone https://github.com/AutonomyLab/bebop_autonomy.git
```
Modify /bebop_driver/src/bebop_video_decoder.cpp
- line 93: CODEC_AP_TRUNCATED -> AV_CODEC_CAP_TRUNCATED
- line 95: CODEC_FLAG_TRUNCATED -> AV_CODEC_FLAG_TRUNCATED
- line 97: CODEC_FLAG2_CHUNKS -> AV_CODEC_FLAG2_CHUNKS

```
code ~/bebop_ws/src/bebop_autonomy/bebop_driver/src/bebop_video_decoder.cpp
```
```
if (codec_ptr_->capabilities & AV_CODEC_CAP_TRUNCATED)
    {
      codec_ctx_ptr_->flags |= AV_CODEC_FLAG_TRUNCATED;
    }
    codec_ctx_ptr_->flags2 |= AV_CODEC_FLAG2_CHUNKS;
```

Add this line in your ~/.bashrc :
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/bebop_ws/devel/lib/parrot_arsdk
```
Install the following:
```
sudo apt install ros-noetic-joy ros-noetic-joy-teleop ros-noetic-teleop-twist-joy
```
Build:
```
cd ..
catkin_make
```

# iROS_drone
```
cd ~/bebop_ws/src
sudo apt-get update
```
```
sudo apt install build-essential python3-rosdep python3-catkin-tools
```
```
sudo apt install libusb-dev python3-osrf-pycommon libspnav-dev libbluetooth-dev libcwiid-dev libgoogle-glog-dev
```
```
sudo apt install ros-noetic-mavros ros-noetic-octomap-ros
```
```
git clone https://github.com/ethz-asl/mav_comm
git clone -b noetic https://github.com/simonernst/iROS_drone
git clone https://github.com/ros-drivers/joystick_drivers
cd ..
catkin_make
```
```
source ~/bebop_ws/devel/setup.bash
```
```
roslaunch rotors_gazebo mav_velocity_control_with_fake_driver.launch
```
# ml2

$ cmake \
-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
-D CUDA_nppicom_LIBRARY=/usr/local/cuda/lib64/libnppicom.so \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D WITH_CUBLAS=ON \
-D WITH_TBB=ON \
-D OPENCV_DNN_CUDA=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D CUDA_ARCH_BIN=7.5 \
-D OPENCV_EXTRA_MODULES_PATH=$HOME/opencv_contrib/modules \
-D BUILD_EXAMPLES=OFF \
-D HAVE_opencv_python3=ON \
..
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
#### 1. Pre-requisites
##### 1.1. CUDA and cuDNN
The main pre-requisite is to have the NVIDIA CUDA Toolkit and the NVIDIA CUDA Deep Neural Network library (cuDNN) library installed.
##### 1.2. Dependencies
CMake and GCC are required to configure the installation and compile the project:
```
sudo apt install cmake
sudo apt install gcc g++
```
To support Python 3 installations, we need Python (it comes pre-installed with most Ubuntu versions), Python-devel and Numpy:
```
sudo apt install python3 python3-dev python3-numpy
```
GTK is requried for GUI features, Camera support (v4l), Media Support (ffmpeg, gstreamer…), etc.:
```
sudo apt install libavcodec-dev libavformat-dev libswscale-dev
sudo apt install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
sudo apt install libgtk-3-dev
```
The next dependencies are optional but add the latest support for PNG, JPEG, JPEG2000, TIFF, WebP, etc. formats:
```
sudo apt install libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev
```
Now, download the OpenCV repository using Git:
```
git clone https://github.com/opencv/opencv.git
```
We will also be downloading the OpenCV’s extra modules (CMake flag -D OPENCV_EXTRA_MODULES_PATH) repository. These modules are required to use the CUDA funcionality with OpenCV.
```
git clone https://github.com/opencv/opencv_contrib.git
```
#### 2. Determine the CUDA architecture version
CMake needs a CUDA_ARCH_BIN flag to be set in order to compile the binaries with the correct CUDA architecture. If this flag is not set correctly, the final use of the binaries would fail.

To set the flag correctly, let’s first determine the NVIDIA GPU model using the nvidia-smi -L command:
Now go to https://developer.nvidia.com/cuda-gpus and look for your GPU model. In my case, I had to go to CUDA-Enabled GeForce and TITAN Products and then find the GeForce GTX 1050:
Next to the model name, you will find the Comput Capability of the GPU. This is the NVIDIA GPU architecture version, which will be the value for the CMake flag: CUDA_ARCH_BIN=6.1.

#### 3. Prepare to compile with CUDA and cuDNN support
We will be using a bunch of CMake flags to compile OpenCV. You can find a detailed reference of these at the end of this article.

Prepare the build directory:
```
cd ~/opencv
mkdir build
cd build
```
Run CMake with the following flags:
```
cmake \
-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
-D CUDA_nppicom_LIBRARY=/usr/local/cuda/lib64/libnppicom.so \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D WITH_CUBLAS=ON \
-D WITH_TBB=ON \
-D OPENCV_DNN_CUDA=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D CUDA_ARCH_BIN=7.5 \
-D OPENCV_EXTRA_MODULES_PATH=$HOME/opencv_contrib/modules \
-D BUILD_EXAMPLES=OFF \
-D HAVE_opencv_python3=ON \
..
```
The \ tell the console to expect a new line inside the same command. The following command is equivalent to that one:
```
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D WITH_CUDNN=ON -D WITH_CUBLAS=ON -D WITH_TBB=ON -D OPENCV_DNN_CUDA=ON -D OPENCV_ENABLE_NONFREE=ON -D CUDA_ARCH_BIN=6.1 -D OPENCV_EXTRA_MODULES_PATH=$HOME/opencv_contrib/modules -D BUILD_EXAMPLES=OFF -D HAVE_opencv_python3=ON ..
```
You should have an output similar to the following one:
```
...
-- The CXX compiler identification is GNU 9.4.0
...
-- Performing Test...
...
-- General configuration for OpenCV 4.5.5-dev ======================
...
--   NVIDIA CUDA:                   YES (ver 11.6, CUFFT CUBLAS)
--     NVIDIA GPU arch:             61
--     NVIDIA PTX archs:
-- 
--   cuDNN:                         YES (ver 8.3.2)
...
-- Configuring done
-- Generating done
-- Build files have been written to: /home/j/opencv/build
That indicates that the pre-build was succesful.
```
#### 4. Compile OpenCV with GPU acceleration support
If CMake exited without any errors, we are ready to compile OpenCV.

To finally compile OpenCV, use make with the -j or --jobs option, which specifies the number of jobs to run simultaneously, and whose value must equal the number of cores we found out previously:
```
make -j$(nproc)
```
In my case it took half an hour to fully complete.

####5. Install OpenCV
After compiling OpenCV with GPU acceleration support through CUDA and cuDNN, we are ready to install it as if we had downloaded a pre-compiled package.

Inside the build directory, run this command:
```
sudo make install
```
Then use ldconfig to create the necessary links and cache to our freshly-installed OpenCV:
```
sudo ldconfig
```
The final step is to create symbolic links to the OpenCV bindings for Python 3, to be used globally. This is required if you use the default Ubuntu installation, as its executable looks for packages in a dist-packages directory, and OpenCV is installed at site-packages by default.

To do this, run the following command:
```
sudo ln -s /usr/local/lib/python3.8/site-packages/cv2 /usr/local/lib/python3.8/dist-packages/cv2
```
Use the python3 interpreter and the cv2.cuda.printCudaDeviceInfo(0) to verify that the library is working correctly:
