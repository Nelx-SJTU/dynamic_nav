# Online Geometric Memory Generation and Maintenance for Visuomotor Navigation in Structure Variant Environments


[**Paper**](https://ieeexplore.ieee.org/document/10387237) 

This repository contains the implementation of the paper:

**Online Geometric Memory Generation and Maintenance for Visuomotor Navigation in Structure Variant Environments**  
Qiming Liu*, Neng Xu*, Zhe Liu and Hesheng Wang (* = Equal contribution)  

We are members in [**IRMV Lab**](https://irmv.sjtu.edu.cn/).

## Installation
First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `visual_nav` using
```
conda env create -n visual_env python=3.8
conda activate visual_nav
```

We provide a requirements file `configs/requirements.txt` to install all the required pakages using
```
cd {YOUR_PROJECT_DIRECTORY}
pip install -r configs/requirements.txt
```

You also need to install the [**iGibson dataset**](https://svl.stanford.edu/igibson/).

## Dataset
You can run the following command to get the RGB image, depth image and pose dataset of Neural Memory Generator in iGibson environment.
```
python mapping/get_dataset.py
```
You can use `'w', 'a', 's', 'd'` to control the robot and use `'o', 'p'` to control the gate in iGibson dataset.

To get the dataset of top-view accessibility map in cartesian coordinate, you can run the following command.
```
python mapping/get_seg_data.py
```

To get the dataset of top-view accessibility map in polar coordinate, you can run the following command.
```
python mapping/img_polar_transform.py
```

## Training
To train the Neural Memory Generator, 
you can run the following command to train:
```
python mapping/cognitive_map_trainer.py
```

To train the Hierarchical Planner, 
you can run the following command to train:
```
python planning/map_planning_trainer.py
```
The supervised dataset of Hierarchical Planner is generated by Astar from `planning/HybridAstarPlanner`

**Note:** You might need to change the directory of models and datasets in these files.

## Navigation Task Example
We provide a sample code for the navigation task. 
First, you need to complete the training of the model and set up the relevant parameters. 
After that, you can run the navigation sample with the following code.
```
python nav_example.py
```

The code is under further development.
