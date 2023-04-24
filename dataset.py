import os
import pandas as pd
import pytorch_lightning as pl
import torch.utils.data
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from thought_object import NumberObject, ConstObject, ThoughtExpander


class DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, datasets, batch_size, test_on_validation, num_token=None, collate_raw=False):
        super(DataModule, self).__init__()
        self.batch_size = batch_size
        if len(datasets) == 3:
            self.train_dataset, self.dev_dataset, self.test_dataset = datasets
            self.predict_dataset = None
        elif len(datasets) == 4:
            self.train_dataset, self.dev_dataset, self.test_dataset, self.predict_dataset = datasets
        self.batch_processor = BatchProcessor(f".language-models/{tokenizer}", num_token=num_token)
        self.test_on_validation = test_on_validation
        if collate_raw:
            self.collate = self.collate_raw
        else:
            self.collate = self.collate_tensor

    @staticmethod
    def as_tensor(k, v):
        if isinstance(v, list) and v:
            if k in ("input_ids", "attention_mask", "label_final"):
                return torch.as_tensor(v)
            elif k in ("label_dd", "label_dd_indices"):
                return [[torch.as_tensor(se) for se in e] for e in v]
        return v

    def as_tensor_batch(self, batch):
        return {k: self.as_tensor(k, v) for k, v in batch.items()}

    def collate_raw(self, batch):
        batch = self.batch_processor(batch)
        return batch

    def collate_tensor(self, batch):
        batch = self.batch_processor(batch)
        return self.as_tensor_batch(batch)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.train_dataset, collate_fn=self.collate, batch_size=self.batch_size, shuffle=True
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(self.test_dataset, collate_fn=self.collate, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.test_on_validation:
            return [
                torch.utils.data.DataLoader(self.dev_dataset, collate_fn=self.collate, batch_size=self.batch_size),
                torch.utils.data.DataLoader(self.test_dataset, collate_fn=self.collate, batch_size=self.batch_size),
            ]
        else:
            return torch.utils.data.DataLoader(self.dev_dataset, collate_fn=self.collate, batch_size=self.batch_size)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(self.predict_dataset, collate_fn=self.collate, batch_size=self.batch_size)


class BatchProcessor:
    def __init__(self, tokenizer_name, num_token):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        assert isinstance(
            self.tokenizer, PreTrainedTokenizerFast
        ), f"{type(self.tokenizer).__name__} is not a fast tokenizer. Slow version of tokenizer is not compatible."
        self.num_token = num_token if num_token is not None else self.tokenizer.mask_token
        self.num_token_id = self.tokenizer.vocab[self.num_token]

    def __call__(self, batch):
        problems = [item["problem"] for item in batch]
        questions = [item["question"] for item in batch]
        tokenized = self.tokenizer(problems, padding="longest", truncation=True, return_offsets_mapping=True)
        offsets = tokenized.pop("offset_mapping")

        tokenized["problem"] = problems
        tokenized["question"] = questions
        tokenized["question_idx"] = [
            next(i for i, (s, e) in enumerate(o) if s <= qe < e)
            for qe, o in zip(((p.index(q) + len(q) - 1) for p, q in zip(problems, questions)), offsets)
        ]
        tokenized["question_indices"] = [
            [i for i, (s, e) in enumerate(o) if qs <= e > 0 and s < qs + qn]
            for (qs, qn), o in zip(((p.index(q), len(q)) for p, q in zip(problems, questions)), offsets)
        ]
        tokenized["num_indices"] = [
            [i for i, token_id in enumerate(ids) if token_id == self.num_token_id] for ids in tokenized["input_ids"]
        ]

        return tokenized | {key: [item[key] for item in batch] for key in batch[0] if key not in tokenized}


class MathDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path,
        tokenizer_name,
        limit_depth,
        compress_num,
        constants,
        ignore_over_depth,
        multi,
        label=False,
        power=None,
        replacer=None,
        num_token=None,
    ):
        data = pd.read_csv(file_path, keep_default_na=False)
        print(f"read {file_path} : n={len(data)}")

        tokenizer = AutoTokenizer.from_pretrained(f".language-models/{tokenizer_name}")
        self.num_token = num_token if num_token is not None else tokenizer.mask_token

        self.multi = multi
        self.compress_num = compress_num
        if constants is None:
            self._extract_constants(data)
        else:
            self.constants = constants
            self.replacer = replacer

        if power is None:
            self.power = any("**" in eq for eq in data["Equation"])
            print(f"This dataset has {'no' if not self.power else ''} power operation.")
        else:
            self.power = power

        self._data = []
        n_ignore, n_unknown_consts = 0, 0
        max_depth = 0
        for problem, question, equation, answer, numbers in zip(
            data["Problem"], data["Question"], data["Equation"], data["Answer"], data["Numbers"]
        ):
            equation = self._fix_equation(equation)
            initial_thoughts = self._generate_initial_thoughts(numbers)
            results = None
            try:
                if multi:
                    results = self._generate_labels(
                        initial_thoughts,
                        equation,
                        limit_depth=limit_depth if ignore_over_depth else -1,
                    )
                if results is None:
                    target = [eval(equation, {b.expr: b for b in initial_thoughts})]
                    for e in target:
                        target += e.children
                    results = self._generate_labels(
                        initial_thoughts,
                        equation,
                        limit_depth=limit_depth if ignore_over_depth else -1,
                        target=target,
                    )
            except NameError:
                if label:
                    n_unknown_consts += 1
            except LookupError as e:
                if ignore_over_depth:
                    n_ignore += 1
                raise e
            if results is None:
                continue
            else:
                (
                    target_thought,
                    label_thoughts,
                    label_indices,
                    label_dds,
                    label_dd_indices,
                    n_thoughts,
                    label_final,
                    n_dds,
                    depth,
                ) = results
                if depth > max_depth:
                    max_depth = depth
            if label:
                example = {
                    "problem": self._process_text(problem),
                    "question": self._process_text(question),
                    "equation": equation,
                    "answer": answer,
                    "initial_thoughts": initial_thoughts,
                    "target_thought": target_thought,
                    "label_thoughts": label_thoughts,
                    "label_thought_indices": label_indices,
                    "label_dd": label_dds,
                    "label_dd_indices": label_dd_indices,
                    "label_final": label_final,
                    "n_thoughts": n_thoughts,
                    "n_dds": n_dds,
                }
            else:
                example = {
                    "problem": self._process_text(problem),
                    "question": self._process_text(question),
                    "equation": equation,
                    "answer": answer,
                    "initial_thoughts": initial_thoughts,
                }
            self._data.append(example)
        if ignore_over_depth and n_ignore:
            print(" - ignored examples :", n_ignore)
        if n_unknown_consts:
            print(" - unknown constants :", n_unknown_consts)
        self.max_depth = max_depth

    def _process_text(self, text):
        if self.num_token != "[NUM]":
            text = text.replace("[NUM]", self.num_token)
        return text

    def _extract_constants(self, raw):
        if "Constants" not in raw:
            print("This dataset does not contain constant values.")
            self.constants, self.replacer = False, False
            return
        collected_consts = [float(c) for c in set(sum((str(c).split() for c in raw["Constants"]), []))]
        replacer = {c: 1 / c for c in collected_consts if c < 1 and (1 / c) in collected_consts}
        self.constants = [ConstObject(c) for c in collected_consts if c not in replacer]
        self.replacer = {ConstObject(k).expr: f"( {(1 / ConstObject(v))} )" for k, v in replacer.items()}
        print("Extracted constants :", [c.value for c in self.constants])

    def _fix_equation(self, equation):
        if self.replacer:
            return " ".join((t if t not in self.replacer else self.replacer[t]) for t in equation.split())
        return equation

    @staticmethod
    def parse_number(number):
        if "%" in number:
            return float(number[:-1]) / 100
        elif "/" in number:
            f1, f2 = number[1:-1].split("/")
            return float(f1) / float(f2)
        return float(number)

    def _generate_initial_thoughts(self, numbers):
        number_thoughts = [NumberObject(i, self.parse_number(n)) for i, n in enumerate(numbers.split())]
        if self.compress_num:
            initial_thoughts = []
            initial_thoughts += (n for n in number_thoughts if not any(n.value == b.value for b in initial_thoughts))
        else:
            initial_thoughts = number_thoughts
        if self.constants:
            initial_thoughts += self.constants
        return initial_thoughts

    def _generate_labels(self, initial_thoughts, equation, *, limit_depth, target=None):
        if target is None:
            target_thought = target = eval(equation, {b.expr: b for b in initial_thoughts})
        else:
            target_thought = target[0] if isinstance(target, list) else target
        label_thoughts, label_indices, label_dds, label_dd_indices = [], [], [], []
        n_dds = 0
        expander = ThoughtExpander(initial_thoughts, limit_depth, power=self.power)
        for expanded_thoughts, expanded_indices in expander:
            label_dd = [float(e in target) for e in expanded_thoughts]
            label_dd_ind = [i for i, s in enumerate(label_dd) if s]

            label_thoughts.append(expanded_thoughts)
            label_indices.append(expanded_indices)
            label_dds.append(label_dd)
            label_dd_indices.append(label_dd_ind)
            n_dds += len(expanded_thoughts)

            expander.collect([expanded_thoughts[i] for i in label_dd_ind])

            final_thought = next((e for e in expanded_thoughts if e == target_thought), None)
            if final_thought is not None:
                label_final = [expander.thoughts.index(final_thought)]
                break
            if len(label_dd_ind) > 30:
                return
        else:
            raise LookupError(
                f"{target_thought!r} is not in expanded thoughts. (equation={equation}, depth={expander.depth - 1})"
            )
        return (
            final_thought,
            label_thoughts,
            label_indices,
            label_dds,
            label_dd_indices,
            len(expander.thoughts),
            label_final,
            n_dds,
            expander.depth,
        )

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


def is_cv(data_root, dataset_name):
    return any(
        not os.path.exists(p)
        for p in (
            os.path.join(data_root, dataset_name, "train.csv"),
            os.path.join(data_root, dataset_name, "dev.csv"),
            os.path.join(data_root, dataset_name, "test.csv"),
        )
    )


def load_datasets(
    data_root,
    dataset_name,
    limit_depth,
    compress_num,
    ignore_over_depth,
    tokenizer=None,
    multi=False,
    label=False,
    power=False,
    num_token=None,
):
    print("Start loading dataset :", dataset_name)
    train_path = os.path.join(data_root, dataset_name, "train.csv")
    dev_path = os.path.join(data_root, dataset_name, "dev.csv")
    test_path = os.path.join(data_root, dataset_name, "test.csv")

    train_dataset = MathDataset(
        train_path,
        tokenizer,
        limit_depth,
        compress_num,
        constants=None,
        ignore_over_depth=ignore_over_depth,
        multi=multi,
        label=True,
        power=power,
        num_token=num_token,
    )
    dev_dataset = MathDataset(
        dev_path,
        tokenizer,
        limit_depth,
        compress_num,
        constants=train_dataset.constants,
        replacer=train_dataset.replacer,
        ignore_over_depth=ignore_over_depth,
        multi=multi,
        label=label,
        power=train_dataset.power,
        num_token=num_token,
    )
    test_dataset = MathDataset(
        test_path,
        tokenizer,
        limit_depth,
        compress_num,
        constants=train_dataset.constants,
        replacer=train_dataset.replacer,
        ignore_over_depth=ignore_over_depth,
        multi=multi,
        label=label,
        power=train_dataset.power,
        num_token=num_token,
    )
    print("maximum test depth :", test_dataset.max_depth)

    return (
        (train_dataset, dev_dataset, test_dataset),
        train_dataset.constants,
        train_dataset.power,
        test_dataset.max_depth,
    )
