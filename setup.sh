conda create -y --prefix ./env python=3.9
conda activate ./env
python -m pip install tensorflow==2.12.0
conda install -y mpi4py
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -e .