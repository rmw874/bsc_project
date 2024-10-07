Log på den der SSH

Download miniforge3 sådan her:

```
$ curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
$ bash Miniforge3-$(uname)-$(uname -m).sh
```

Også få den til at installere det her bs:

```
$ conda create --name mgr python=3.10
$ conda activate mgr
$ conda install -c pytorch -c nvidia -c conda-forge -c defaults \
    colorama \
    cudatoolkit=11.8 \
    numba \
    matplotlib \
    openblas \
    pandas \
    scikit-image \
    numpy \
    scikit-learn \
    scipy \
    tqdm \
    wandb \
    wheel \
    ipython \
    pytorch::pytorch=2.2 \
    pytorch::pytorch-cuda=12.1
$ pip install jupyter-client jupyter-core jupyterlab randomname kornia opencv-python-headless joblib pillow albumentations torchmetrics adabelief-pytorch segmentation-models-pytorch
