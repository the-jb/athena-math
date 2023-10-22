import os
import platform
from typing import Any, Optional

import pytorch_lightning as pl
import yaml
from pytorch_lightning.utilities.types import STEP_OUTPUT

try:
    import fcntl

    def lock_file(f):
        fcntl.lockf(f, fcntl.LOCK_EX)

    def unlock_file(f):
        f.flush()
        os.fsync(f.fileno())
        fcntl.lockf(f, fcntl.LOCK_UN)

except ModuleNotFoundError:

    def lock_file(f):
        pass

    def unlock_file(f):
        pass


class ValidateTestMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_val_score = None

        self.val_score = None
        self.val_corrects = None
        self.n_vals = None

        self.test_score = None
        self.test_corrects = None
        self.n_tests = None

        self.do_test = None

    def is_best(self, score):
        if self.best_val_score is None:
            return True
        return score > self.best_val_score

    def _on_validation_epoch_start(self):
        self.n_vals = 0
        self.val_corrects = 0
        self.do_test = None
        self.on_real_validation_epoch_start()

    def _on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        if batch is None:
            return

        if batch_idx == 0 and dataloader_idx == 1:
            if self.end_validation():
                self.start_test()

    def _on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        if batch is None:
            return

        if dataloader_idx == 0:
            self.end_validation_batch(outputs, batch, batch_idx)
        elif dataloader_idx == 1 and self.do_test:
            self.end_test_batch(outputs, batch, batch_idx)

    def end_validation_batch(self, outputs, batch, batch_idx):
        predictions, corrects = outputs
        self.val_corrects += sum(corrects)
        self.n_vals += len(predictions)

        self.on_real_validation_batch_end(outputs, batch, batch_idx)

    def end_test_batch(self, outputs, batch, batch_idx):
        predictions, corrects = outputs
        self.test_corrects += sum(corrects)
        self.n_tests += len(predictions)

        self.on_real_test_batch_end(outputs, batch, batch_idx)

    def _on_validation_epoch_end(self):
        if self.do_test:
            self.end_test()
        else:
            self.end_validation()

    def end_validation(self):
        if self.n_vals == 0:
            return

        self.val_score = self.val_corrects / self.n_vals * 100
        updated = self.update_best_val_score(self.val_score)
        self.on_real_validation_epoch_end()
        return updated

    def update_best_val_score(self, score):
        if self.is_best(score):
            self.best_val_score = score
            self.on_update_best_validation_score()
            return True
        return False

    def start_test(self):
        self.do_test = True
        self.n_tests = 0
        self.test_corrects = 0
        self.on_real_test_epoch_start()

    def end_test(self):
        if self.n_tests == 0:
            return

        self.test_score = self.test_corrects / self.n_tests * 100
        self.on_real_test_epoch_end()

    def on_real_validation_epoch_start(self):
        pass

    def on_real_validation_batch_end(self, outputs, batch, batch_idx: int):
        pass

    def on_real_validation_epoch_end(self):
        pass

    def on_update_best_validation_score(self):
        pass

    def on_real_test_epoch_start(self):
        pass

    def on_real_test_batch_end(self, outputs, batch, batch_idx: int):
        pass

    def on_real_test_epoch_end(self):
        pass


class ValidateTestHook(ValidateTestMixin):
    def on_validation_epoch_start(self):
        super()._on_validation_epoch_start()

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        super()._on_validation_batch_start(batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        super()._on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self):
        super()._on_validation_epoch_end()

    def on_test_epoch_start(self):
        super().start_test()

    def on_test_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int = 0):
        super().end_test_batch(outputs, batch, batch_idx)

    def on_test_epoch_end(self):
        super().end_test()


class ValidateTestCallback(ValidateTestMixin, pl.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.current_epoch = None

    def on_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.current_epoch = trainer.current_epoch

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        super()._on_validation_epoch_start()

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        super()._on_validation_batch_start(batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        super()._on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        super()._on_validation_epoch_end()

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        super().start_test()

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        super().end_test()

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        super().end_test_batch(outputs, batch, batch_idx)


class BestScoreSummary(ValidateTestCallback):
    DEFAULT_FILENAME = "result"

    def __init__(
        self, filename, val_keys, test_keys, node=None, save_dir="results", force_filename=False
    ):
        super().__init__()
        print(f"Summary : {filename=} {val_keys=} {test_keys=} {node=}")
        self.val_keys = [val_keys] if isinstance(val_keys, str) else val_keys
        self.test_keys = [test_keys] if isinstance(test_keys, str) else test_keys

        self.node = platform.node() if node is None else node

        self._init_path(filename, save_dir, force_filename)
        self.best_val_epoch = None

    def _format_file(self, filename):
        return f"{filename}.{self.node}.yaml"

    def _init_path(self, name, save_dir, force_filename):
        os.makedirs(save_dir, exist_ok=True)
        if name is None:
            name = self.DEFAULT_FILENAME

        filename = name
        if not force_filename:
            version_cnt = 0
            while os.path.exists(os.path.join(save_dir, self._format_file(filename))):
                version_cnt += 1
                filename = f"{name}.{version_cnt:02d}"

        self.filename = filename
        print("Summary filename :", filename)
        self._path = os.path.join(save_dir, self._format_file(filename))

    def on_update_best_validation_score(self):
        self.update_result(
            f"{self.best_val_score} (epoch={self.current_epoch})", keys=self.val_keys
        )

    def on_real_test_epoch_end(self):
        self.update_result(f"{self.test_score} (epoch={self.current_epoch})", keys=self.test_keys)

    def update_result(self, result, keys):
        if os.path.exists(self._path):
            with open(self._path, "r+") as f:
                lock_file(f)
                results = yaml.load(f, yaml.FullLoader)
                unlock_file(f)
            if results is None:
                results = {}
        else:
            results = {}
        target = results
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            if not isinstance(target[key], dict):
                if not isinstance(target[key], list):
                    target[key] = [target[key], {}]
                target = target[key][1]
            else:
                target = target[key]
        key = keys[-1]
        if key not in target:
            target[key] = result
        elif isinstance(target[key], list):
            target[key][0] = result
        elif isinstance(target[key], dict):
            target[key] = [result, target[key]]
        else:
            target[key] = result

        with open(self._path, "w") as f:
            lock_file(f)
            yaml.dump(results, f)
            unlock_file(f)


class LogEvaluation(ValidateTestCallback):
    def __init__(self, filename, train_setting, save_dir="outputs", node=None):
        super().__init__()
        print(f"LogTest : {filename=}")
        self.best_score = None

        self.node = platform.node() if node is None else node
        self.train_setting = train_setting

        self.valid_path = self._init_path(f"{self.node}.valid.{filename}", save_dir)
        self.test_path = self._init_path(f"{self.node}.test.{filename}", save_dir)
        print("Log filename :", self.valid_path, self.test_path)
        self.outputs = []

    @staticmethod
    def _format_file(filename):
        for c in '<>\\:"/|?*':
            filename = filename.replace(c, "")
        return f"{filename}.txt"

    def _init_path(self, filename, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        version_cnt = 0
        path = os.path.join(save_dir, self._format_file(filename))
        while True:
            try:
                open(path, "x").close()
                break
            except FileExistsError:
                path = os.path.join(
                    save_dir,
                    self._format_file(f"{filename}.{(version_cnt := version_cnt + 1):02d}"),
                )

        return path

    def on_real_validation_epoch_start(self):
        self.outputs = []

    def on_update_best_validation_score(self):
        self.write_outputs(
            self.valid_path,
            f"Epoch:{self.current_epoch} val_score:{self.best_val_score} {self.train_setting}",
        )

    def on_real_validation_batch_end(self, outputs, batch, batch_idx: int):
        self.append_outputs(batch, outputs)

    def on_real_test_epoch_start(self):
        self.outputs = []

    def on_real_test_epoch_end(self):
        self.write_outputs(
            self.test_path,
            f"Epoch:{self.current_epoch} val_score:{self.best_val_score} test_score:{self.test_score} {self.train_setting}",
        )

    def on_real_test_batch_end(self, outputs, batch, batch_idx: int):
        self.append_outputs(batch, outputs)

    def append_outputs(self, batch, outputs):
        predictions, corrects = outputs
        self.outputs += [
            (problem, equation, prediction.expr, correct)
            for problem, equation, prediction, correct in zip(
                batch["problem"], batch["equation"], predictions, corrects
            )
        ]

    def write_outputs(self, path, info):
        with open(path, "w") as f:
            lock_file(f)
            f.write(f"{info}\n\n")
            for i, (problem, equation, expression, correct) in enumerate(self.outputs):
                f.write(f"{i + 1:03}. {problem}\n")
                f.write(f"solution : {equation}\n")
                f.write(f"predict  : {expression} ({'O' if correct else 'X'})\n\n")
            unlock_file(f)
