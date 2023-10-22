import pytorch_lightning as pl
import torch.optim
from transformers import AutoModel, PreTrainedModel

from callbacks import ValidateTestHook

from .athena import Athena


class LitAthena(ValidateTestHook, pl.LightningModule):
    def __init__(
        self,
        language_model,
        p_drop,
        ln_type,
        strength,
        hidden_size,
        ff_size,
        n_heads,
        max_depth,
        chain,
        goal,
        ref,
        reason,
        threshold,
        lr_scheduler,
        lr,
        weight_decay,
        lr_factor,
        lr_step_size,
        lm_lr,
        epoch,
        start_validation_loss,
        constants,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.language_model: PreTrainedModel = AutoModel.from_pretrained(
            f".language-models/{language_model}"
        )
        lm_hidden_size = self.language_model.config.hidden_size
        if n_heads is None:
            n_heads = self.language_model.config.num_attention_heads
        if ff_size is None:
            ff_size = self.language_model.config.intermediate_size

        self.athena = Athena(
            max_depth,
            constants,
            lm_hidden_size,
            hidden_size,
            n_heads,
            ff_size,
            p_drop,
            ln_type,
            strength,
            chain,
            goal,
            ref,
            reason,
        )
        self.threshold = threshold

        self.lr_scheduler = lr_scheduler
        self.lr = lr
        self.lm_lr = lm_lr
        self.weight_decay = weight_decay
        self.lr_factor = lr_factor
        self.lr_step_size = lr_step_size
        self.target_epoch = epoch
        self.start_validation_loss = start_validation_loss
        self._skip_validation = False
        self._skip_test_epoch = False

    def configure_optimizers(self):
        if self.lm_lr is None:
            params = self.parameters()
        else:
            params = [
                {"params": self.language_model.parameters(), "lr": self.lm_lr},
                {"params": self.athena.parameters()},
            ]

        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, self.lr_step_size, self.lr_factor
            )
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1, end_factor=self.lr_factor, total_iters=self.target_epoch
            )
        return [optimizer], [scheduler]

    def forward(self, batch):
        x = self.language_model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )[0]

        x = self.athena(batch, x, self.threshold)
        return x

    @staticmethod
    def measure_answer(prediction, answer):
        if prediction is None:
            return False
        result = prediction.value

        s1, s2 = str(result), str(answer)
        n_float = min(len(s1[s1.find(".") :]) - 1, len(s2[s2.find(".") :]) - 1, 4)
        return abs(result - answer) < 10**-n_float

    def measure(self, predictions, batch):
        corrects = [
            self.measure_answer(prediction, answer)
            for prediction, answer in zip(predictions, batch["answer"])
        ]
        return corrects

    def training_step(self, batch, batch_idx):
        predictions, loss = self(batch)
        self.log("train_loss", loss, on_epoch=True, on_step=False, batch_size=len(predictions))

        return loss

    def transfer_batch_to_device(self, batch, device: torch.device, dataloader_idx: int = 0):
        if not self.training:
            if self._skip_validation:
                return
            elif dataloader_idx == 1 and self.do_test is False:
                return
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def on_validation_start(self):
        if "train_loss" in self.trainer.callback_metrics:
            self._skip_validation = (
                0 < self.start_validation_loss < self.trainer.callback_metrics["train_loss"]
            )
        else:
            self._skip_validation = self.start_validation_loss > 0

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if batch is None:
            return

        predictions, attn_score = self(batch)
        corrects = self.measure(predictions, batch)
        accuracy = sum(corrects) / len(predictions)
        batch_size = len(predictions)

        prefix = "val" if dataloader_idx == 0 else "test"
        self.log(
            f"{prefix}/accuracy",
            accuracy,
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=batch_size,
        )
        if attn_score is not None:
            predictions = [
                LitAthena._ScoreObject(pred, score) for pred, score in zip(predictions, attn_score)
            ]
        return predictions, corrects

    def on_real_validation_epoch_end(self):
        self.log("score", self.val_score, prog_bar=True, add_dataloader_idx=False)

    def test_step(self, batch, batch_idx):
        predictions, attn_score = self(batch)

        corrects = self.measure(predictions, batch)
        return predictions, corrects

    def test_epoch_end(self, outputs):
        n = 0
        n_corrects = 0
        for predictions, corrects in outputs:
            n += len(predictions)
            n_corrects += sum(corrects)
        accuracy = n_corrects * 100 / n
        self.log("test_score", accuracy, prog_bar=True, add_dataloader_idx=False, batch_size=n)
        return accuracy

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        predictions, attn_score = self(batch)
        return [(prediction.expr, prediction.value, attn_score) for prediction in predictions]

    class _ScoreObject:
        def __init__(self, p, s):
            self.expr = p.expr + "\n"
            for k, v in s.items():
                self.expr += f"{k} = {v}\n"
