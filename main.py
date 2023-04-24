import os
import yaml
import fire
import pytorch_lightning as pl
import pytorch_lightning.loggers
from models.lit_athena import LitAthena
from callbacks import BestScoreSummary, LogEvaluation
from dataset import DataModule, load_datasets, is_cv


def _train_cv(cv, dataset, fold_start, fold_end, score_filename, node, force_filename, **train_kwargs):
    score_filename = BestScoreSummary(score_filename, None, None, node=node, force_filename=force_filename).filename
    for i in range(fold_start, fold_end):
        print(f"Fold {i + 1}/5 Start")
        train(
            cv=False,
            dataset=os.path.join(dataset, f"fold{i}"),
            score_filename=score_filename,
            node=node,
            force_filename=True,
            **train_kwargs,
        )


def train(
    epoch=100,
    batch_size=4,
    dataset="asdiv-a",
    cv=False,
    fold_start=0,
    fold_end=5,
    seed=None,
    gpu=-1,
    log=True,
    ignore_over_depth=False,
    multi=False,
    log_path="logs",
    ckpt_path="ckpts",
    ckpt=False,
    score_filename=None,
    node=None,
    force_filename=True,
    language_model="roberta-base",
    num_token=None,
    compress_num=False,
    p_drop=0.5,
    ln="pre",
    strength=0.95,
    hidden_size=None,
    ff_size=None,
    n_heads=None,
    limit_depth=19,
    chain=1,
    goal=0,
    ref=True,
    reason=True,
    threshold=0.5,
    swa=0.3,
    lr=1.3e-5,
    weight_decay=1e-5,
    lr_scheduler="step",
    lr_factor=0.7,
    lr_step_size=20,
    lm_lr=None,
    start_validation_loss=0,
    test_on_validation=True,
    skip_test=True,
):
    if score_filename is None:
        score_filename = f"{language_model}.{dataset}"
        force_filename = False
    if cv or is_cv("data", dataset):
        return _train_cv(**locals())
    pl.seed_everything(seed)

    if not os.path.exists(os.path.join(".language-models", language_model)):
        if "/" in language_model:
            print(f"Download {language_model}. Use language-model={language_model.split('/')[-1]} for further runs.")
        language_model = download(language_model)

    ckpt = not log and ckpt

    datasets, constants, has_power, max_depth = load_datasets(
        data_root="data",
        dataset_name=dataset,
        limit_depth=limit_depth,
        compress_num=compress_num,
        ignore_over_depth=ignore_over_depth,
        tokenizer=language_model,
        multi=multi,
        num_token=num_token,
    )
    datamodule = DataModule(
        tokenizer=language_model,
        datasets=datasets,
        batch_size=batch_size,
        test_on_validation=test_on_validation,
        num_token=num_token,
    )
    model = LitAthena(
        language_model=language_model,
        p_drop=p_drop,
        ln_type=ln,
        strength=strength,
        hidden_size=hidden_size,
        ff_size=ff_size,
        n_heads=n_heads,
        max_depth=max_depth,
        chain=chain,
        goal=goal,
        ref=ref,
        reason=reason,
        threshold=threshold,
        lr_scheduler=lr_scheduler,
        lr=lr,
        weight_decay=weight_decay,
        lr_factor=lr_factor,
        lr_step_size=lr_step_size,
        lm_lr=lm_lr,
        epoch=epoch,
        start_validation_loss=start_validation_loss,
        constants=constants,
        has_power=has_power,
        data_path="data",
        dataset=dataset,
        batch_size=batch_size,
        collate_raw=False,
    )

    if log:
        model_name = f"{language_model}-{ln=!s}"
        if hidden_size is not None:
            model_name += f"-{hidden_size=}"
        if ff_size is not None:
            model_name += f"-{ff_size=}"
        if n_heads is not None:
            model_name += f"-{n_heads=}"
        train_setting = (
            f"{batch_size=}-{epoch=}-{p_drop=}"
            f"-{lr_scheduler}[{lr_step_size},{lr_factor}]-{lr=}-{lm_lr=}-{weight_decay=}-{swa=}"
            f"-{threshold=}"
        )

        dataset_name_keys = dataset.split(os.sep)
        dataset_name = f"{seed=}-" + "-".join(dataset_name_keys)
        depth = f"{max_depth=} " + ("(multi)" if multi else "(single)")

        logger = [
            pl.loggers.TensorBoardLogger(
                log_path, name=model_name, version=train_setting, sub_dir=dataset_name, default_hp_metric=False
            ),
            pl.loggers.CSVLogger(log_path, name=model_name, version=train_setting + os.sep + dataset_name),
        ]
        callbacks = [
            pl.callbacks.LearningRateMonitor(),
            BestScoreSummary(
                filename=score_filename,
                node=node,
                val_keys=[
                    model_name,
                    f"{dataset_name_keys[0]} (validation)",
                    train_setting,
                    f"{seed=}",
                    *dataset_name_keys[1:],
                ],
                test_keys=[
                    model_name,
                    f"{dataset_name_keys[0]} (test)",
                    train_setting,
                    f"{seed=}",
                    *dataset_name_keys[1:],
                ],
                force_filename=force_filename,
            ),
            LogEvaluation(filename=f"{model_name}.{dataset_name}", train_setting=train_setting, node=node),
        ]
        if ckpt:
            ckpt_path = logger[0].log_dir.replace(log_path, ckpt_path)
            ckpt_path, filename = os.path.split(ckpt_path)
            ckpt_callback = pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_path,
                filename=f"{filename}-{{epoch:02d}}-{{score:.5f}}",
                monitor="score",
                save_top_k=1,
                mode="max",
                save_last=False,
            )
            callbacks.append(ckpt_callback)
    else:
        logger = False
        callbacks = []

    if swa:
        callbacks.append(pl.callbacks.StochasticWeightAveraging(swa_epoch_start=epoch - swa if swa >= 1 else 1 - swa))

    trainer = pl.Trainer(
        max_epochs=epoch,
        logger=logger,
        gpus=[gpu] if gpu >= 0 else -1,
        enable_checkpointing=ckpt,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        auto_select_gpus=gpu < 0,
    )
    trainer.fit(model, datamodule=datamodule)
    if not skip_test:
        if ckpt:
            print("Test with best checkpoint")
            trainer.test(ckpt_path="best", datamodule=datamodule)
        else:
            print("Test with last model")
            trainer.test(model, datamodule=datamodule)


def test(model_path, hparam_path, dataset):
    with open(hparam_path, "r") as fp:
        hparams = yaml.load(fp, Loader=yaml.UnsafeLoader)

    model = LitAthena.load_from_checkpoint(model_path, skip_validation=5, lr=None, hparams_file=hparam_path)
    datasets, constants, has_power = load_datasets(
        data_root="data",
        dataset_name=dataset,
        limit_depth=hparams["limit_depth"],
        compress_num=hparams["compress_num"],
        ignore_over_depth=True,
    )
    datamodule = DataModule(
        tokenizer=hparams["language_model"], datasets=datasets, batch_size=32, test_on_validation=False
    )

    trainer = pl.Trainer(logger=False, enable_checkpointing=False)
    results = trainer.test(model, datamodule=datamodule)

    acc = 0
    n = 0
    corrects, incorrects = [], []
    for c, ic, n_batch in results:
        acc += len(c)
        n += n_batch
        corrects += c
        incorrects += ic
    print("Correct:")
    for correct in corrects:
        for k, v in correct.items():
            print(f"{k} : {v}")
        print()
    print("Incorrect:")
    for incorrect in incorrects:
        for k, v in incorrect.items():
            print(f"{k} : {v}")
        print()
    print("Final Accuracy : ", acc / n)


def inspect_data(
    dataset="asdiv-a/fold0",
    data_path="data",
    tokenizer="roberta-base",
    compress_num=False,
    limit_depth=19,
    power=False,
    ignore_over_depth=False,
    multi=False,
):
    datasets, constants, has_power, max_depth = load_datasets(
        data_path,
        dataset,
        limit_depth,
        compress_num,
        ignore_over_depth,
        tokenizer=tokenizer,
        multi=multi,
        label=True,
        power=power,
    )
    from collections import Counter
    from statistics import mean, stdev

    def stderr(data):
        return stdev(data) / len(data) ** 0.5

    def _print_statistics(name, datas):
        for mode, data in zip(("(train)", "  (dev)", " (test)", "  (all)"), datas):
            print(
                f"{name} {mode} : {max(data)=}",
                f"{min(data)=}",
                f"{mean(data)=:.2f}+-{stderr(data):.2f}" f"{stdev(data)=:.2f}" f"{Counter(data)}",
                sep=", ",
            )
        print()

    def _print_lists(name, datas):
        for mode, data in zip(("(train)", "  (dev)", " (test)", "  (all)"), datas):
            print(
                f"{name} {mode} : {max(data)=}"
                f"{min(data)=}"
                f"{mean(data)=:.2f}+-{stderr(data):.2f}"
                f"{sorted(data, reverse=True)[:200]}",
                sep=", ",
            )
        print()

    def _print_values(name, values):
        for mode, value in zip(("(train)", "  (dev)", " (test)", "  (all)"), values):
            print(f"{name} {mode} : {value}")
        print()

    def _n_thoughts(data):
        ns = [[] for _ in range(max(len(d["label_thoughts"]) for d in data))]
        for d in data:
            for i, es in enumerate(d["label_thoughts"]):
                ns[i].append(len(es))
        return "\n " + "\n ".join(
            f"{i:>2}: {len(n)=:5}"
            + (f", {max(n)=:5}, {min(n)=:2}, {mean(n)=:3.2f}+-{stderr(data):.2f}" if len(n) > 1 else f", {n=}")
            for i, n in enumerate(ns)
            if len(n) > 0
        )

    datamodule = DataModule(tokenizer, datasets, batch_size=1, collate_raw=True, test_on_validation=False)

    trd = datamodule.train_dataloader()
    vd = datamodule.val_dataloader()
    tsd = datamodule.test_dataloader()
    trd = [{k: v[0] for k, v in d.items()} for d in trd]
    vd = [{k: v[0] for k, v in d.items()} for d in vd]
    tsd = [{k: v[0] for k, v in d.items()} for d in tsd]
    ad = (trd + vd) if len(vd) == len(tsd) else (trd + vd + tsd)
    _print_values("n_equation", (len(Counter(d["equation"] for d in data)) for data in (trd, vd, tsd, ad)))
    _print_statistics(
        "n_operations",
        (
            [sum(d["equation"].replace("**", "^").count(op) for op in ["+", "-", "*", "/", "^"]) for d in data]
            for data in (trd, vd, tsd, ad)
        ),
    )
    _print_statistics("target depth", ([len(d["label_thoughts"]) for d in data] for data in (trd, vd, tsd, ad)))
    _print_values("n_thoughts per depth", (_n_thoughts(data) for data in (trd, vd, tsd)))
    _print_lists("last depth n_thoughts", ([len(d["label_thoughts"][-1]) for d in data] for data in (trd, vd, tsd, ad)))
    _print_lists("n_dds", ([d["n_dds"] for d in data] for data in (trd, vd, tsd, ad)))
    _print_lists("n_thoughts", ([d["n_thoughts"] for d in data] for data in (trd, vd, tsd, ad)))
    _print_values(
        "max(n_thoughts per depth)",
        (
            max(max(len(e) for e in d["label_thoughts"]) for d in data if len(d["label_thoughts"]) > 0)
            for data in (trd, vd, tsd)
        ),
    )


def download(language_model="roberta-base", num_token=None):
    print(f"download() : {language_model=}")
    from transformers import AutoModel, AutoTokenizer, BertTokenizer
    from tokenizers import AddedToken

    try:
        tokenizer = AutoTokenizer.from_pretrained(language_model)
    except:
        tokenizer = BertTokenizer.from_pretrained(language_model)

    plm = AutoModel.from_pretrained(language_model)
    if num_token is not None:
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    AddedToken(num_token, single_word=False, lstrip=True, rstrip=False, normalized=False)
                ]
            }
        )
        plm.resize_token_embeddings(plm.config.vocab_size + 1)

    language_model = language_model.split("/")[-1].lower()
    path = f".language-models/{language_model}"

    tokenizer.save_pretrained(path)
    plm.save_pretrained(path)
    return language_model


if __name__ == "__main__":
    fire.Fire()
