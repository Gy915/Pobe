# Pobe(On going...)

## DATA

data can be found in `.\dataset`

## Requirements

* PyTorch == 1.8.1
* transformers == 4.17.0
* faiss-gpu == 1.7.2
* python >= 3.7

## Usage

Take **IMDB-CLINC**  pair as example. 

* Finetuning Stage

```shell
python main.py --dataset IMDB
```

​	After finetuning, put `index_path`,  `token_prefix` in the crossponding position in `config/pobe.yaml`.

* Inference Stage

```shell
python Pobe.py --K 1024
```

