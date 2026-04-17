## Environment

- Ubuntu 22.04

- CUDA 11.8

- Python 3.7

- PyTorch 1.13.1 + cu117

  

## Installation

Install the environment.

```
conda create -n pointrm python=3.7
conda activate pointrm
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install -r requirements.txt

# chamfer distance & emd
cd ./extensions/chamfer_dist && python setup.py install --user
cd ../..
cd ./extensions/emd && python setup.py install --user
cd ../..

pip install --upgrade pip setuptools
pip install --upgrade pip
```

Install the PointNet++.

```
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

Install the GPU kNN.

```
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

Download and install the causal-conv1d.

```
https://github.com/Dao-AILab/causal-conv1d/releases

pip install causal_conv1d-1.1.1+cu118torch1.13cxx11abiFALSE-cp37-cp37m-linux_x86_64.whl
```

Download and install the mamba.

```
https://github.com/state-spaces/mamba/releases
python -m pip install safetensors==0.4.3
pip install mamba_ssm-1.1.1+cu118torch1.13cxx11abiFALSE-cp37-cp37m-linux_x86_64.whl
```



## Classification on ModelNet40

### Dataset

Download the processed ModelNet data from https://cloud.tsinghua.edu.cn/d/4808a242b60c4c1f9bed/. 

```
data
 |--- ModelNet
       |--- modelnet40_normal_resampled
             |--- modelnet40_shape_names.txt
             |--- modelnet40_train.txt
             |--- modelnet40_test.txt
             |--- modelnet40_train_8192pts_fps.dat
             |--- modelnet40_test_8192pts_fps.dat
```

### Training

Training from scratch.

```
CUDA_VISIBLE_DEVICES=0 python main.py --scratch_model --config cfgs/finetune_modelnet_2024.yaml --exp_name exp_2024
```

Saved logs and models.

```
logs
 |--- exp_2024
       |--- 2024.log
       |--- ckpt-best.pth
       |--- ckpt-last.pth
       |--- config.yaml
 |--- TFBoard
       |--- exp_2024
             |--- train
             |--- test
```



## Classification on ScanObjectNN

### Dataset

Download the processed h5_files.zip from https://hkust-vgd.github.io/scanobjectnn/. 

```
data
 |--- ScanObjectNN
       |--- main_split
             |--- test_objectdataset.h5
             |--- test_objectdataset_augmentedrot_scale75.h5
             |--- ... ...
             |--- training_objectdataset.h5
             |--- training_objectdataset_augmentedrot_scale75.h5
             |--- ... ...
       |--- main_split_nobg
       |--- split1
       |--- split1_nobg
       |--- split2
       |--- split2_nobg
       |--- split3
       |--- split3_nobg
       |--- split4
       |--- split4_nobg
```

### Training

Training from scratch.

```
CUDA_VISIBLE_DEVICES=0 python main.py --scratch_model --config cfgs/finetune_scan_objbg.yaml --exp_name objbg_2024

CUDA_VISIBLE_DEVICES=0 python main.py --scratch_model --config cfgs/finetune_scan_objonly.yaml --exp_name objonly_2024

CUDA_VISIBLE_DEVICES=0 python main.py --scratch_model --config cfgs/finetune_scan_hardest.yaml --exp_name hardest_2024
```

Saved logs and models.

```
logs
 |--- objbg_2024
       |--- 2024.log
       |--- ckpt-best.pth
       |--- ckpt-last.pth
       |--- config.yaml
 |--- TFBoard
       |--- objbg_2024
```



## Part Segmentaion on ShapeNetPart

### Dataset

Download dataset from https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip. 

```
data
 |--- shapenetcore_partanno_segmentation_benchmark_v0_normal
       |--- 02691156
             |--- .txt
       |--- ... ...
       |--- 04379243
             |--- .txt
       |--- train_test_split
       |--- synsetoffset2category.txt
```

### Training

Delete the line 205 and 207 in the 'part_segmentation/main.py'.

```
scheduler = CosineLRScheduler(optimizer,
                              t_initial=sche_config.kwargs.epochs,
                              lr_min=1e-6,
                              warmup_lr_init=1e-6,
                              warmup_t=sche_config.kwargs.initial_epochs,
                              cycle_limit=1,
                              t_in_epochs=True)
```

Training from scratch.

```
cd part_segmentation

CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/config.yaml --log_dir snp_2024
```

Saved logs and models.

```
part_segmentation
 |--- part_seg
       |--- snp_2024
             |--- checkpoints
                   |--- best_model.pth
             |--- logs
                   |--- pt_mamba.txt
             |--- pointrm.py
```
