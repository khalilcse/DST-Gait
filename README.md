# DST-Gait
This repository contains the PyTorch Lighting Code for gait recognition. The part of code is borrowed from other projects. Thanks for their wonderful work!
- GaitGraph2: [tteepe/GaitGraph2](https://github.com/tteepe/GaitGraph2)
- st-gcn: [yysijie/st-gcn](https://github.com/yysijie/st-gcn)

## Preparation
Clone the repository and install the dependencies from `requirements.txt`.

### Datasets
It is recommended to go to download the dataset and follow the website's instructions to get the password to run the repository.
- CASIA-B: [http://www.cbsr.ia.ac.cn/china/Gait%20Databases%20CH.asp](http://www.cbsr.ia.ac.cn/china/Gait%20Databases%20CH.asp) or
Download from [here](https://github.com/tteepe/GaitGraph/releases/tag/v0.1) and move `casia-b_pose_coco.csv` to `data`
- OUMVLP-Pose: [http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitLPPose.html](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitLPPose.html)

## Running the code
We use [PyTorch Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html) for configuration and training.

Train:
```bash
# CASIA-B
python3 DSTGait_casia_b.py fit --config configs/casia_b.yaml 
# OUMVLP-Pose
python3 DSTGait_oumvlp.py fit --config configs/oumvlp.yaml

```
Test:
```bash
python3 DSTGait_{casia_b,oumvlp}.py predict --config <path_to_config_file> --ckpt_path <path_to_checkpoint> --model.tta False
```
Logs and checkpoints will be saved to `lighting_logs` and can be shown in tensorboard with:
```bash
tensorboard --logdir lightning_logs
```

## Visualization of CASIA-B Sample for NM#, BG#, and CL# Conditions

CASIA-B Sample Example:<br>
![001](CASIAB-01.gif)
