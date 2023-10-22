from collections import Counter
from statistics import mean, stdev

from dataset import DataModule, load_datasets


def inspect_data(
    dataset="asdiv-a/fold0",
    data_path="data",
    tokenizer="roberta-base",
    compress_num=False,
    limit_depth=19,
    ignore_over_depth=False,
    multi=False,
):
    datasets, *_ = load_datasets(
        data_path,
        dataset,
        limit_depth,
        compress_num,
        ignore_over_depth,
        tokenizer=tokenizer,
        multi=multi,
        label=True,
    )

    datamodule = DataModule(
        tokenizer,
        datasets,
        batch_size=1,
        collate_raw=True,
        test_on_validation=False,
    )

    trd = datamodule.train_dataloader()
    vd = datamodule.val_dataloader()
    tsd = datamodule.test_dataloader()
    trd = [{k: v[0] for k, v in d.items()} for d in trd]
    vd = [{k: v[0] for k, v in d.items()} for d in vd]
    tsd = [{k: v[0] for k, v in d.items()} for d in tsd]
    ad = (trd + vd) if len(vd) == len(tsd) else (trd + vd + tsd)

    print_values(
        "n_equation",
        (len(Counter(d["equation"] for d in data)) for data in (trd, vd, tsd, ad)),
    )
    print_statistics(
        "n_operations",
        (
            [
                sum(d["equation"].replace("**", "^").count(op) for op in ["+", "-", "*", "/", "^"])
                for d in data
            ]
            for data in (trd, vd, tsd, ad)
        ),
    )
    print_statistics(
        "target depth",
        ([len(d["label_thoughts"]) for d in data] for data in (trd, vd, tsd, ad)),
    )
    print_values(
        "n_thoughts per depth",
        (get_n_thoughts(data) for data in (trd, vd, tsd)),
    )
    print_lists(
        "last depth n_thoughts",
        ([len(d["label_thoughts"][-1]) for d in data] for data in (trd, vd, tsd, ad)),
    )
    print_lists(
        "n_dds",
        ([d["n_dds"] for d in data] for data in (trd, vd, tsd, ad)),
    )
    print_lists(
        "n_thoughts",
        ([d["n_thoughts"] for d in data] for data in (trd, vd, tsd, ad)),
    )
    print_values(
        "max(n_thoughts per depth)",
        (
            max(
                max(len(e) for e in d["label_thoughts"])
                for d in data
                if len(d["label_thoughts"]) > 0
            )
            for data in (trd, vd, tsd)
        ),
    )


def stderr(data):
    return stdev(data) / len(data) ** 0.5


def print_statistics(name, datas):
    for mode, data in zip(("(train)", "  (dev)", " (test)", "  (all)"), datas):
        print(
            f"{name} {mode} : {max(data)=}",
            f"{min(data)=}",
            f"{mean(data)=:.2f}+-{stderr(data):.2f}" f"{stdev(data)=:.2f}" f"{Counter(data)}",
            sep=", ",
        )
    print()


def print_lists(name, datas):
    for mode, data in zip(("(train)", "  (dev)", " (test)", "  (all)"), datas):
        print(
            f"{name} {mode} : {max(data)=}"
            f"{min(data)=}"
            f"{mean(data)=:.2f}+-{stderr(data):.2f}"
            f"{sorted(data, reverse=True)[:200]}",
            sep=", ",
        )
    print()


def print_values(name, values):
    for mode, value in zip(("(train)", "  (dev)", " (test)", "  (all)"), values):
        print(f"{name} {mode} : {value}")
    print()


def get_n_thoughts(data):
    ns = [[] for _ in range(max(len(d["label_thoughts"]) for d in data))]
    for d in data:
        for i, es in enumerate(d["label_thoughts"]):
            ns[i].append(len(es))
    return "\n " + "\n ".join(
        f"{i:>2}: {len(n)=:5}"
        + (
            f", {max(n)=:5}, {min(n)=:2}, {mean(n)=:3.2f}+-{stderr(data):.2f}"
            if len(n) > 1
            else f", {n=}"
        )
        for i, n in enumerate(ns)
        if len(n) > 0
    )
