# Installation

This project is tested on python 3.10 with torch 2.3.0.
To run code in this project a gpu-version torch is preferred.

Users can install necessary dependencies by `pip install -r requirements.txt`

# Usage

We use [hydra](https://github.com/facebookresearch/hydra/) to manage experiments and configurations.
Please refer to [hydra doc](https://hydra.cc/) for more details on how to tweak each configuration. However 
below we have listed the most important commands to run.

## Step I. Train the model
Training model or finding the optimal architecture is not the goal of this project, thus we leverage the two widely-used existing architectures namely
[DDIM](https://arxiv.org/abs/2010.02502) and [PF](https://arxiv.org/abs/2011.13456).


User can train the DDIM model by the following commands
```python
# on MNIST
python run.py train=optim
# on KMNIST
python run.py train=optim dataset.dataset=KMNIST
# on FMNIST
python run.py train=optim dataset.dataset=FMNIST
```
In `cfg/train/optim.yaml` user can optimizer's hyperparamters. 
The default checkpoint save directory is defined in `save_path` and user can change it to desired path.

Alternatively, to train score-based model (for probability flow ODE) user can add the following flag:
```python
python run.py train=optim dataset.dataset=FMNIST model=PF
```

All related training scripts are in the `train.sh`. Training takes roughly one hour for each model/config. 
We also provided the trained checkpoint [here](https://huggingface.co/datasets/cnut1648/DDIM_and_PF_on_MNIST_FMNIST_KMNIST) so that you can skip this section.

## Step II. To sample from model
User can again use hydra to launch inference with different sampling methods 
```
# on MNIST
python run.py sample.model_path="Your CKPT Path" sample_speed=50 method="F-PNDM"
# on KMNIST
python run.py dataset.dataset=KMNIST sample.model_path="Your CKPT Path"
# on FMNIST
python run.py dataset.dataset=FMNIST sample.model_path="Your CKPT Path"
```
where:
- `sample.model_path` is the path to pretrained model ckpt. This is the ckpt you obtained from Step I.
- `sample_speed` here faster speed means more sample frequency, i.e. inference would be a bit slower
- `method` determine the decoding strategy, in this project we experiment "F-PNDM", "PF", "PR-CR", "DDIM", "S-PNDM" and "SP-PNDM".

All related sampling scripts are in the `infer.sh`. Results will be saved in `inference_results/` folder.

## Step III. FID Evaluation

`inference_results/` contains three subfolders: MNIST, KMNIST, FMNIST.

In each of the subfolders there are `<model>-<sample method>`: DDIM-DDIM  DDIM-F-PNDM  DDIM-PF  DDIM-PR-CR  DDIM-S-PNDM  PF-DDIM  PF-F-PNDM  PF-PF  PF-PR-CR  PF-S-PNDM.

Then in each of these nested subfolder, there are three folders: 600, 800, 1000 i.e. different diffusion step (i.e. steps in solving SDE). Within these folders are the 101 sampled images named 0.png to 100.png. 

We use `run_FID.py` to run FID calculation on these images. It will create `fid_results.json` in `FMNIST/`, `KMNIST/`, `MNIST/` folders.

Lastly, you can use `viz_FID.py` to visualize and produce the plot we show in the writeup.

All results in this step can also be found [here](https://huggingface.co/datasets/cnut1648/DDIM_and_PF_on_MNIST_FMNIST_KMNIST/tree/main/inference_results).