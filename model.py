import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import LightningModule
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

class BertModel(LightningModule):
    
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained(self.hparams.model_name, num_labels=2)
    
    def forward(self, input_ids, attention_mask, labels):
        x = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return x 

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch

        outputs = self.forward(input_ids, attention_mask, labels)

        loss = outputs[0]
        logits = torch.max(outputs[1], dim=1)[1]

        logits = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        score = f1_score(y_true=labels, y_pred=logits)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_score", score, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": logits, "labels": labels}
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch

        outputs = self.forward(input_ids, attention_mask, labels)

        loss = outputs[0]
        logits = torch.max(outputs[1], dim=1)[1]

        logits = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        score = f1_score(y_true=labels, y_pred=logits)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_score", score, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": logits, "labels": labels}

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            self.hparams.lr
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.hparams.total_steps
        )

        return {"optimizer": optimizer, "scheduler": scheduler}
