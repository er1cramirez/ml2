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
