# HOTFormerLoc: Hierarchical Octree Transformer for Versatile Lidar Place Recognition Across Ground and Aerial Views

### What's new ###
* [2025-XX-XX] Training and evaluation code released. CS-Wild-Places dataset released.

## Description
This is the official repository for the paper:

**HOTFormerLoc: Hierarchical Octree Transformer for Versatile Lidar Place Recognition Across Ground and Aerial Views**, CVPR 2025 [[arXiv](https://github.com/csiro-robotics/HOTFormerLoc)] by *Ethan Griffiths, Maryam Haghighat, Simon Denman, Clinton Fookes, and Milad Ramezani*

![Network Architecture](media/hotformerloc_architecture.png)


We present HOTFormerLoc, a novel and versatile **H**ierarchical **O**ctree-based **T**rans**Former** for large-scale 3D place recognition. We propose an octree-based multi-scale attention mechanism that captures spatial and semantic features across granularities, making it suitable for both ground-to-ground and ground-to-aerial scenarios across urban and forest environments.

<!-- <img src="media/radar_plot.svg" alt="Hero Figure" width="50%" height="auto" style="float: right;"> -->

In addition, we introduce our novel dataset: CS-Wild-Places, a 3D cross-source dataset featuring point cloud data from aerial and ground lidar scans captured in dense forests. Point clouds in CS-Wild-Places contain representational gaps and distinctive attributes such as varying point densities and noise patterns, making it a challenging benchmark for cross-view localisation in the wild.

Our results demonstrate that HOTFormerLoc achieves a top-1 average recall improvement of 5.5% – 11.5% on the CS-Wild-Places benchmark. Furthermore, it consistently outperforms SOTA 3D place recognition methods, with an average performance gain of 5.8% on well established urban and forest datasets. 

<!-- ![Hero Figure](media/radar_plot.svg) -->
<img src="media/radar_plot.svg" alt="Hero Figure" width="50%" height="auto" style="display: block; margin: auto;">

### Citation
If you find this work useful, please consider citing:
```
@InProceedings{HOTFormerLoc,
	author    = {Griffiths, Ethan and Haghighat, Maryam and Denman, Simon and Fookes, Clinton and Ramezani, Milad},
	title     = {HOTFormerLoc: Hierarchical Octree Transformer for Versatile Lidar Place Recognition Across Ground and Aerial Views},
	booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	month     = {todo},
	year      = {2025},
	pages     = {todo}
}
```

## Environment and Dependencies
Code was tested using Python 3.11 with PyTorch 2.1.1 and CUDA 12.1 on a Linux system. We use conda to manage dependencies (although we recommend [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) for a much faster install).

### Installation
```
# Note: replace 'mamba' with 'conda' if using a vanilla conda install
mamba create -n hotformerloc python=3.11 -c conda-forge
mamba activate hotformerloc
mamba install 'numpy<2.0' -c conda-forge
mamba install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge
pip install -r requirements.txt
pip install libs/dwconv
```

Modify the `PYTHONPATH` environment variable to include the absolute path to the project root folder (ensure this variable is set every time you open a new shell): 
```export PYTHONPATH
export PYTHONPATH=$PYTHONPATH:<path/to/repo>
```

## Datasets

### Wild-Places
*Info about dataset here + link to download and instructions for use*

Place or symlink the data in `data/wild_places`.

Run the following to fix the broken timestamps in the poses files:
```
cd datasets/WildPlaces
python fix_broken_timestamps.py \
	--root '../../data/wild_places/data/' \
	--csv_filename 'poses.csv' \
	--csv_savename 'poses_fixed.csv' \
	--cloud_folder 'Clouds'

python fix_broken_timestamps.py \
	--root '../../data/wild_places/data/' \
	--csv_filename 'poses_aligned.csv' \
	--csv_savename 'poses_aligned_fixed.csv' \
	--cloud_folder 'Clouds_downsampled'
```

Before network training or evaluation, run the below code to generate pickles with positive and negative point clouds for each anchor point cloud:
```
cd datasets/WildPlaces
python generate_training_tuples.py \
	--root '../../data/wild_places/data/' 

python generate_test_sets.py \
	--root '../../data/wild_places/data/'
```

### CS-Wild-Places
*Info about dataset here + link to download and instructions for use, including Wild-Places setup*

1. Download
2. Place or symlink the data in `data/CS-Wild-Places`

Assuming you have followed the above instructions to install Wild-Places (generating train/test tuples is not required), you can use the below command to postprocess the Wild-Places ground submaps into the format required for CS-Wild-Places (set num_workers to a sensible number for your system). Note that this may take several hours depending on your CPU:
```
cd datasets/CSWildPlaces
python postprocess_wildplaces_ground.py \
	--root '../../data/wild_places/data/' \
	--cswildplaces_save_dir '../../data/CS-Wild-Places/postproc_voxel_0.80m_rmground_normalised' \
	--remove_ground \
	--downsample \
	--downsample_type 'voxel' \
	--voxel_size 0.8 \
	--normalise \
	--num_workers XX \
	--verbose
```
Note that this script will generate the submaps used for the results reported in the paper, i.e. voxel downsampled, ground points removed, and normalised. We also provide a set of unnormalised submaps for convenience, and the corresponding Wild-Places ground submaps can be generated by omitting the `--normalise` option, and by setting the `--cswildplaces_save_dir` to `'../../data/CS-Wild-Places/postproc_voxel_0.80m_rmground'`.

Before network training or evaluation, run the below code to generate pickles with positive and negative point clouds for each anchor point cloud:

```
cd datasets/CSWildPlaces
python generate_train_test_tuples.py \
	--root '../../data/CS-Wild-Places/postproc_voxel_0.80m_rmground_normalised/' \
	--eval_thresh '30' \
	--pos_thresh '15' \
	--neg_thresh '60' \
	--buffer_thresh '30' \
	--v2_only
```
Note that training and evaluation pickles are saved to the directory specified in `--root`. 

### CS-Campus3D
We train on the CS-Campus3D dataset introduced in *CrossLoc3D: Aerial-Ground Cross-Source 3D Place Recognition* ([link](https://arxiv.org/pdf/2303.17778)).

Download the dataset [here](https://drive.google.com/file/d/1yxVicykRMg_HAfZG2EQUl1R3_wxpxStd/view?usp=sharing), and place or symlink the data in the `data/benchmark_datasets_cs_campus3d` directory. 

*Add instructions to fix the tuple format (below should work but verify this)*

```
cd datasets/CSCampus3D
python save_queries_HOTFormerLoc_format.py
```

### Oxford RobotCar
We trained on a subset of Oxford RobotCar and the In-house (U.S., R.A., B.D.) datasets introduced in
*PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition* ([link](https://arxiv.org/pdf/1804.03492)).
There are two training datasets:
- Baseline Dataset - consists of a training subset of Oxford RobotCar
- Refined Dataset - consists of training subset of Oxford RobotCar and training subset of In-house

We report results on the Baseline set in the paper.

For dataset description see the PointNetVLAD paper or github repository ([link](https://github.com/mikacuy/pointnetvlad)).

You can download training and evaluation datasets from 
[here](https://drive.google.com/open?id=1rflmyfZ1v9cGGH0RL4qXRrKhg-8A-U9q) 
([alternative link](https://drive.google.com/file/d/1-1HA9Etw2PpZ8zHd3cjrfiZa8xzbp41J/view?usp=sharing)). 

Place or symlink the data in `data/benchmark_datasets`.

Before network training or evaluation, run the below code to generate pickles with positive and negative point clouds for each anchor point cloud. 

```generate pickles
cd datasets/pointnetvlad 

# Generate training tuples for the Baseline Dataset
python generate_training_tuples_baseline.py --dataset_root '../../data/benchmark_datasets'

# (Optionally) Generate training tuples for the Refined Dataset
python generate_training_tuples_refine.py --dataset_root '../../data/benchmark_datasets'

# Generate evaluation tuples
python generate_test_sets.py --dataset_root '../../data/benchmark_datasets'
```

## Training
To train **HOTFormerLoc**, download and extract the datasets and generate training pickles as described above, for any dataset you wish to train on. 
The configuration files for each dataset can be found in `config/`. 
Set `dataset_folder` parameter to the dataset root folder.
If running out of GPU memory, decrease `batch_split_size` and `val_batch_size` parameter value. If running out of RAM, you may need to decrease the `batch_size` parameter, but note this may slightly reduce performance. We use wandb for logging by default, but this can be disabled in the config.

To train the network, run:

```
cd training

# To train HOTFormerLoc on CS-Wild-Places
python train.py --config ../config/config_cs-wild-places.txt --model_config ../models/hotformerloc_cs-wild-places_cfg.txt

# To train HOTFormerLoc on Wild-Places
python train.py --config ../config/config_wild-places.txt --model_config ../models/hotformerloc_wild-places_cfg.txt

# To train HOTFormerLoc on CS-Campus3D
python train.py --config ../config/config_cs-campus3d.txt --model_config ../models/hotformerloc_cs-campus3d_cfg.txt

# To train HOTFormerLoc on Oxford RobotCar
python train.py --config ../config/config_oxford.txt --model_config ../models/hotformerloc_oxford_cfg.txt
```

### Pre-trained Models

Pretrained models are available in `weights` directory
- `hotformerloc_cs-wild-places.pth` trained on the CS-Wild-Places Dataset 
- `hotformerloc_wild-places.pth` trained on the Wild-Places Dataset 
- `hotformerloc_cs-campus3d.pth` trained on the CS-Campus3D Dataset 
- `hotformerloc_oxford.pth` trained on the Oxford RobotCar Dataset 

### Evaluation

To evaluate the pretrained models run the following commands:

```
cd eval

# To evaluate HOTFormerLoc trained on CS-Wild-Places
python pnv_evaluate.py --config ../config/config_cs-wild-places.txt --model_config ../models/hotformerloc_cs-wild-places_cfg.txt --weights ../weights/hotformerloc_cs-wild-places.pth

# To evaluate HOTFormerLoc trained on Wild-Places
python pnv_evaluate.py --config ../config/config_wild-places.txt --model_config ../models/hotformerloc_wild-places_cfg.txt --weights ../weights/hotformerloc_wild-places.pth

# To evaluate HOTFormerLoc trained on CS-Campus3D
python pnv_evaluate.py --config ../config/config_cs-campus3d.txt --model_config ../models/hotformerloc_cs-campus3d_cfg.txt --weights ../weights/hotformerloc_cs-campus3d.pth

# To evaluate HOTFormerLoc trained on Oxford RobotCar
python pnv_evaluate.py --config ../config/config_oxford.txt --model_config ../models/hotformerloc_oxford_cfg.txt --weights ../weights/hotformerloc_oxford.pth
```

## Results (TO-DO)

**MinkLoc3Dv2** performance (measured by Average Recall@1) compared to the state of the art:

### Trained on Baseline Dataset

| Method                   | Oxford     | U.S.       | R.A.       | B.D        | Average    |
|--------------------------|------------|------------|------------|------------|------------|
| PointNetVLAD [1]         | 62.8       | 63.2       | 56.1       | 57.2       | 59.8       |
| PCAN [2]                 | 69.1       | 62.4       | 56.9       | 58.1       | 61.6       |
| LPD-Net [4]              | 86.3       | 87.0       | 83.1       | 82.5       | 94.7       |
| EPC-Net [5]              | 86.2       | -          | -          | -          | -          | 
| NDT-Transformer [7]      | 93.8       | -          | -          | -          | -          |
| MinkLoc3D [8]            | 93.0       | 86.7       | 80.4       | 81.5       | 85.4       |
| PPT-Net [9]              | 93.5       | 90.1       | 84.1       | 84.6       | 88.1       |
| SVT-Net [10]             | 93.7       | 90.1       | 84.4       | 85.5       | 88.4       |
| TransLoc3D [11]          | 95.0       | -          | -          | -          | -          |
| ***MinkLoc3Dv2 (ours)*** | ***96.3*** | ***90.9*** | ***86.5*** | ***86.3*** | ***90.0*** |

### Acknowledgements

Special thanks to the authors of [MinkLoc3Dv2](https://github.com/jac99/MinkLoc3Dv2) and [OctFormer](https://github.com/octree-nn/octformer) for their excellent code, which formed the foundation of this codebase. 