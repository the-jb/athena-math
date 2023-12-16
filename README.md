# ATHENA for Mathematical Reasoning

**ATHENA** (**A**ttention-based **TH**ought **E**xpansion **N**etwork **A**rchitecture) is a neural architecture of thought expansion for mathematical reasoning.

Link to our [Conference paper](https://aclanthology.org/2023.emnlp-main.1014/) [arXiv version](https://arxiv.org/abs/2311.01036)

<p align=center>
<img src="figures/athena_fig1.png" style="width: 50%; height: 50%;" />
</p>

## Results

| Model                      | MAWPS | ASDIV-A | Math23k | SVAMP | UnbiasedMWP | SVAMP (1:N) | UnbiasedMWP (1:N) |
|----------------------------|-------|---------|---------|-------|-------------|-------------|-------------------|
| **ATHENA (RoBERTa-base)**  | 92.2  | 86.4    | 84.4    | 45.6  | 36.2        | 52.5        | 35.4              |
| **ATHENA (RoBERTa-large)** | 93.0  | 91.0    | 86.5    | 54.8  | 42.0        | 67.8        | 48.4              |



## Datasets

Our repository includes the following datasets:

- [MAWPS](https://aclanthology.org/N16-1136)
- [ASDiv](https://aclanthology.org/2020.acl-main.92)
- [SVAMP](https://aclanthology.org/2021.naacl-main.168)
- [Math23k](https://aclanthology.org/D17-1088/)  


## Run Model
### Install Environments
- Python: 3.9
- Requirements
```
pip install -r requirements.txt
```

### Pre-download Language Models
We recommend preparing language models before training.
```
python main.py download --language-model=roberta-base
```

### Training
We offer various options for training.
The list of arguments can be found in [`train()`](https://github.com/the-jb/athena-math/blob/main/main.py#L32).

The following shows a few examples:

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
> The code does support distributed training with multiple GPUs.

### Default path

- Dataset: `data`
- Logs: `logs`
- Checkpoints: `ckpts`
- Results (for the best score record): `results`
- Outputs (for the prediction outputs): `outputs`
- Language Models: `.language-models`

### Test models

```
python main.py test --model-path="ckpts/<ckpt_path>/<ckpt_filename>.ckpt" \
                    --hparam-path="logs/<log_path>/hparams.yaml" \
                    --dataset=svamp
```

### Inspect Dataset

```
python main.py inspect-data --dataset=cv_asdiv-a/fold0
```

# Citation
```
@inproceedings{kim-etal-2023-athena,
    title = "{ATHENA}: Mathematical Reasoning with Thought Expansion",
    author = "Kim, JB.  and
      Kim, Hazel  and
      Hahn, Joonghyuk  and
      Han, Yo-Sub",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.1014",
    pages = "16315--16327",

}
```
