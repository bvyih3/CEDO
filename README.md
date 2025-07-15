# CEDO
PyTorch code for IJCAI 2025 paper "Cause-Effect Driven Optimization for Robust Medical Visual Question Answering with Language Biases".  
  
# Requirements
* python 3.7.11
* pytorch 1.10.2+cu113
* torchvision 0.11.3+cu113

# Installation
```bash
conda create -n CEDO python=3.7
conda activate CEDO
pip install -r requirements.txt
```

## Data Setup
keep file in the folders set by `utils/config.py`.
The new constructed dataset `slake-cp` and `vqa-rad-cp` are provided in `data/`.

## Preprocessing

The preprocessing steps are as follows:

1. process questions and dump dictionary:
    ```
    python tools/create_dictionary.py
    ```

2. process answers and question types, and generate the frequency-based margins:
    ```
    python tools/compute_softscore.py
    ```
3. convert image features to h5:
    ```
    python tools/detection_features_converter.py 
    ```

## Model training instruction
```
    python main.py --name test-VQA --gpu 0 --dataset DATASET
   ```
Set `DATASET` to a specfic dataset such as `slake`, `slake-cp`, `vqa-rad`, and `vqa-rad-cp`. 

## Model evaluation instruction
```
    python main_arcface.py --name test-VQA --test
   ```

# Citation
Please cite our paper if you find them helpful :)

```

```
