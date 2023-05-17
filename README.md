# ALTO: Alternating Latent Topologies for Implicit 3D Reconstruction
[**Paper**](https://arxiv.org/abs/2212.04096) | [**Project Page**](https://visual.ee.ucla.edu/alto.htm/) <br>

<div style="text-align: center">
<img src="media/pipeline.png" width="800"/>
</div>

This repository contains the implementation of the paper:

ALTO: Alternating Latent Topologies for Implicit 3D Reconstruction

If you find our code or paper useful, please consider citing
```bibtex
@inproceedings{Wang2023CVPR,
    title = {ALTO: Alternating Latent Topologies for Implicit 3D Reconstruction},
    author = {Wang, Zhen and Zhou, Shijie and Park, Jeong Joon and Paschalidou, Despoina and You, Suya and Wetzstein, Gordon and Guibas, Leonidas and Kadambi, Achuta},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year = {2023}
}
```

## Installation
You can create an anaconda environment called `alto` using
```
conda env create -f environment.yaml
conda activate alto
```
**Note**: you might need to install **torch-scatter** mannually following [the official instruction](https://github.com/rusty1s/pytorch_scatter#pytorch-140):
```
pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
```

Next, compile the extension modules.
You can do this via
```
python setup.py build_ext --inplace
```

## Dataset
In this paper, we consider 3 different datasets:
### Synthetic Indoor Scene Dataset
You can download the preprocessed data (144 GB) using

```
bash scripts/download_data.sh
```

This script should download and unpack the data automatically into the `data/synthetic_room_dataset` folder.  

### ShapeNet
You can download the dataset (73.4 GB) by running the [script](https://github.com/autonomousvision/occupancy_networks#preprocessed-data) from Occupancy Networks. After, you should have the dataset in `data/ShapeNet` folder.

### ScanNet
Download ScanNet v2 data from the [official ScanNet website](https://github.com/ScanNet/ScanNet).
Then, you can preprocess data with:
`scripts/dataset_scannet/build_dataset.py` and put into `data/ScanNet` folder.  

## Experiments
### Training
To train a network, run:
```
python train.py CONFIG.yaml
```
For available training options, please take a look at `configs/default.yaml`.

**Note**: We implement the code in a multiple-GPU version. Please make sure to call the right version of our encoder at `Line 99` for feature triplane or `Line 100` for feature volume in `train.py`.

### Mesh Generation
To generate meshes using a trained model, use
```
python generate.py CONFIG.yaml
```
where you replace `CONFIG.yaml` with the correct config file.


### Evaluation
For evaluation of the models, we provide the script `eval_meshes.py`. You can run it using:
```
python eval_meshes.py CONFIG.yaml
```
The script takes the meshes generated in the previous step and evaluates them using a standardized protocol. The output will be written to `.pkl/.csv` files in the corresponding generation folder which can be processed using [pandas](https://pandas.pydata.org/).

### Acknowledgement 
The code is largely based on [ConvONet](https://github.com/autonomousvision/convolutional_occupancy_networks). Many thanks to the authors for opensourcing the codebase. 

---
## Pretrained models

[ShapeNet 3k](https://drive.google.com/file/d/17AHuISu1f8xWQFevQ2K1lWE1A4LlmsI2/view?usp=share_link)

[Synthetic Room 10k](https://drive.google.com/file/d/1cffLRxa6mGZlMuUwTEKrMJIxyfC7MDJu/view?usp=share_link)
