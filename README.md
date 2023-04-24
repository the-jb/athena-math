# ATHENA

**ATHENA** (**A**ttention-based **TH**ought **E**xpansion **N**etwork **A**rchitecture) is a neural architecture of thought expansion that yields reasonable thoughts for mathematical reasoning.

Link to our [paper]()

<p align=center>
<img src="figures/athena_fig1.png" style="width: 50%; height: 50%;" />
</p>

## Results

| Model                      | MAWPS | ASDIV-A | Math23k | SVAMP | UnbiasedMWP | SVAMP (1:N) | UnbiasedMWP (1:N) |
|----------------------------|-------|---------|---------|-------|-------------|-------------|-------------------|
| **ATHENA (RoBERTa-base)**  | 92.2  | 86.4    | 84.4    | 45.6  | 36.2        | 52.5        | 35.4              |
| **ATHENA (RoBERTa-large)** | 93.0  | 91.0    | 86.5    | 54.8  | 42.0        | 67.8        | 48.4              |



## Datasets

Our repository includes with following datasets:

- [MAWPS](https://aclanthology.org/N16-1136)
- [ASDiv](https://aclanthology.org/2020.acl-main.92)
- [SVAMP](https://aclanthology.org/2021.naacl-main.168)
- [Math23k](https://aclanthology.org/D17-1088/)  
- [MathQA](https://aclanthology.org/N19-1245/)


## Run Models
### Setting Environments
- Python : 3.9
- Requirements
```
pip install -r requirements.txt
```

### Dataset

- Default dataset folder : `data`

### Training

- Training dataset
```
python main.py train --dataset=asdiv-a
```

- Training with fixed seed
```
python main.py train --dataset=asdiv-a --seed=100
```

- Training with specific GPU
```
python main.py train --gpu=0 --dataset=asdiv-a
```
> Note that our model architecture cannot support distributed training with multiple GPUs.

- Other arguments for models can be found in `train()` in `main.py`

### Result files

- Default log_path : `logs`
- Default checkpoint path : `ckpts`
- Default result path (saving best score record) : `results`
- Default output path: `outputs`

### Test models

```
python main.py test --model-path="ckpts/<ckpt_path>/<ckpt_filename>.ckpt" \
                    --hparam-path="logs/<log_path>/hparams.yaml" \
                    --dataset=svamp
```

### Other features

- Inspect dataset
```
python main.py inspect-data --dataset=cv_asdiv-a/fold0
```

# Citation

```

```
