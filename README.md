# STORM
```
conda create -n storm python=3.10
conda activate storm

# for gpu
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# for cpu
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

pip install -r requirements.txt
```

## install apex
For performance and full functionality, we recommend installing Apex with CUDA and C++ extensions via
```
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
APEX also supports a Python-only build via

```
pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./
```


# Start

## 01. Pretraining (Predicton and Portfio Management)
```
accelerate launch --main_process_port 29507 tools/train.py --config configs/exp/pretrain_day_dj30_dynamic_dual_vqvae.py
```

## 02. Extract State for RL
```
accelerate launch --main_process_port 29600 tools/train.py --no_train --no_test --state --no_writer --no_wandb --config configs/state/pretrain_day_dj30_dynamic_dual_vqvae.py
```

## 03. RL for trading
```
CUDA_VISIBLE_DEVICES=0 python tools/trading.py --config=configs/agent/AAPL_day_dj30_dynamic_dual_vqvae.py
```