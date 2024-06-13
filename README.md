## IND-WhoIsWho

## Team DeepMayNotLearn	Rank 11

## Prerequisites

- Ubuntu22.04
- Python 3.10
- PyTorch 2.2.0+cu121

## Hardware device

- GPU 3090 24G
- RAM 64G

## Run Code

### Run GNN

install the requirements

```sh
sh GNN/install_requirements.sh
```

build the graph ,and this may cost lots of time,if need you can find me to get my built graph

```shell
sh GNN/build_graph.sh
```

train and save the model

```shell
sh GNN/train_model.sh
```

predict

```
python GNN/w2v_model_test.py

python GNN/sci_model_test.py
```

### Run LGB+XGB+CAT

```sh
sh lgb+cat+xgb/install_requires.sh
```

train_w2v
```
python w2vProces.py
```

train-and-predict
```
python lgb+cat+xgb/main.py
```

## Resourece

IND Dataset:[Data (biendata.xyz)](https://www.biendata.xyz/competition/ind_kdd_2024/data/)

SciBERT model:[allenai/scibert_scivocab_uncased at main (hf-mirror.com)](https://hf-mirror.com/allenai/scibert_scivocab_uncased/tree/main)

## Methods

You can see more details in GNN/README.md and lgb+cat+xgb/README.md



