# STORM
We propose a Spatio-Temporal factOR Model based on dual vector quantized
variational autoencoders, named STORM, which extracts features of
stocks from temporal and spatial perspectives, then fuses and aligns
these features at the fine-grained and semantic level, and represents
the factors as multi-dimensional embeddings. The discrete code-
books cluster similar factor embeddings, ensuring orthogonality
and diversity, which helps distinguish between different factors
and enables factor selection in financial trading.

## Installation
### Prepare environment
```
conda create -n storm python=3.10
conda activate storm

# for gpu
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# for cpu
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

pip install -r requirements.txt
```

### Install apex (optional)
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


# Running

## Data Preparation
```
# download data
python tools/download.py --config configs/download/dj30.py

# preprocess data
python tools/data_preprocess.py --config configs/processor/processor_day_dj30.py
```

## Pretraining
In our implementation, the `prediction` task and the downstream `portfolio management` task are integrated. As a result, we can compute the metrics for both tasks during the pretraining phase.
```
# exmpales
# only train
accelerate launch tools/pretrain_dynamic_dual_vqvae.py --train --no_test

# only test
accelerate launch tools/pretrain_dynamic_dual_vqvae.py --no_train --test --no_tensorboard --no_wandb

# train and test
# 29507, 29508, 29509, 29510 are the main_process_port. You can change it to other numbers. But make sure they are different.
accelerate launch --main_process_port 29507 tools/pretrain_dynamic_dual_vqvae.py --config configs/exp/pretrain/pretrain_day_dj30_dynamic_dual_vqvae.py
accelerate launch --main_process_port 29508 tools/pretrain_dynamic_dual_vqvae.py --config configs/exp/pretrain/pretrain_day_dj30_dynamic_single_vqvae_time_series.py
accelerate launch --main_process_port 29509 tools/pretrain_dynamic_dual_vqvae.py --config configs/exp/pretrain/pretrain_day_dj30_dynamic_single_vqvae_cross_sectional.py
accelerate launch --main_process_port 29510 tools/pretrain_dynamic_dual_vqvae.py --config configs/exp/pretrain/pretrain_day_dj30_dynamic_single_vqvae_mix.py

accelerate launch --main_process_port 29511 tools/pretrain_dynamic_dual_vqvae.py --config configs/exp/pretrain/pretrain_day_sp500_dynamic_dual_vqvae.py
accelerate launch --main_process_port 29512 tools/pretrain_dynamic_dual_vqvae.py --config configs/exp/pretrain/pretrain_day_sp500_dynamic_single_vqvae_time_series.py
accelerate launch --main_process_port 29513 tools/pretrain_dynamic_dual_vqvae.py --config configs/exp/pretrain/pretrain_day_sp500_dynamic_single_vqvae_cross_sectional.py
accelerate launch --main_process_port 29514 tools/pretrain_dynamic_dual_vqvae.py --config configs/exp/pretrain/pretrain_day_sp500_dynamic_single_vqvae_mix.py
```

## Trading
Since our downstream task can also be a trading task implemented via RL, we first need to extract factor vectors as the state before training the RL agent for trading.
DJ30 is used as an example here. You can replace it with other datasets.
```
# extract state
accelerate launch --main_process_port 29600 tools/pretrain_dynamic_dual_vqvae.py --no_train --no_test --state --no_tensorboard --no_wandb --config configs/exp/state/state_day_dj30_dynamic_dual_vqvae.py --checkpoint_path workdir/pretrain_day_dj30_dynamic_dual_vqvae/checkpoint/best.pth
accelerate launch --main_process_port 29600 tools/pretrain_dynamic_dual_vqvae.py --no_train --no_test --state --no_tensorboard --no_wandb --config configs/exp/state/state_day_dj30_dynamic_single_vqvae_cross_sectional.py --checkpoint_path workdir/pretrain_day_dj30_dynamic_single_vqvae_cross_sectional/checkpoint/best.pth
accelerate launch --main_process_port 29600 tools/pretrain_dynamic_dual_vqvae.py --no_train --no_test --state --no_tensorboard --no_wandb --config configs/exp/state/state_day_dj30_dynamic_single_vqvae_time_series.py --checkpoint_path workdir/pretrain_day_dj30_dynamic_single_vqvae_time_series/checkpoint/best.pth

# trading
# CUDA_VISIBLE_DEVICES=0 is optional, you can remove it if you don't want to specify the GPU.
CUDA_VISIBLE_DEVICES=0 python tools/trading_dynamic_dual_vqvae.py --config=configs/exp/trading/trading_AAPL_day_dj30_dynamic_dual_vqvae.py
CUDA_VISIBLE_DEVICES=0 python tools/trading_dynamic_dual_vqvae.py --config=configs/exp/trading/trading_IBM_day_dj30_dynamic_dual_vqvae.py
CUDA_VISIBLE_DEVICES=0 python tools/trading_dynamic_dual_vqvae.py --config=configs/exp/trading/trading_INTC_day_dj30_dynamic_dual_vqvae.py
CUDA_VISIBLE_DEVICES=0 python tools/trading_dynamic_dual_vqvae.py --config=configs/exp/trading/trading_JPM_day_dj30_dynamic_dual_vqvae.py
CUDA_VISIBLE_DEVICES=0 python tools/trading_dynamic_dual_vqvae.py --config=configs/exp/trading/trading_MSFT_day_dj30_dynamic_dual_vqvae.py

CUDA_VISIBLE_DEVICES=0 python tools/trading_dynamic_dual_vqvae.py --config=configs/exp/trading/trading_AAPL_day_dj30_dynamic_single_vqvae_cross_sectional.py
CUDA_VISIBLE_DEVICES=0 python tools/trading_dynamic_dual_vqvae.py --config=configs/exp/trading/trading_IBM_day_dj30_dynamic_single_vqvae_cross_sectional.py
CUDA_VISIBLE_DEVICES=0 python tools/trading_dynamic_dual_vqvae.py --config=configs/exp/trading/trading_INTC_day_dj30_dynamic_single_vqvae_cross_sectional.py
CUDA_VISIBLE_DEVICES=0 python tools/trading_dynamic_dual_vqvae.py --config=configs/exp/trading/trading_JPM_day_dj30_dynamic_single_vqvae_cross_sectional.py
CUDA_VISIBLE_DEVICES=0 python tools/trading_dynamic_dual_vqvae.py --config=configs/exp/trading/trading_MSFT_day_dj30_dynamic_single_vqvae_cross_sectional.py

CUDA_VISIBLE_DEVICES=0 python tools/trading_dynamic_dual_vqvae.py --config=configs/exp/trading/trading_AAPL_day_dj30_dynamic_single_vqvae_time_series.py
CUDA_VISIBLE_DEVICES=0 python tools/trading_dynamic_dual_vqvae.py --config=configs/exp/trading/trading_IBM_day_dj30_dynamic_single_vqvae_time_series.py
CUDA_VISIBLE_DEVICES=0 python tools/trading_dynamic_dual_vqvae.py --config=configs/exp/trading/trading_INTC_day_dj30_dynamic_single_vqvae_time_series.py
CUDA_VISIBLE_DEVICES=0 python tools/trading_dynamic_dual_vqvae.py --config=configs/exp/trading/trading_JPM_day_dj30_dynamic_single_vqvae_time_series.py
CUDA_VISIBLE_DEVICES=0 python tools/trading_dynamic_dual_vqvae.py --config=configs/exp/trading/trading_MSFT_day_dj30_dynamic_single_vqvae_time_series.py
```