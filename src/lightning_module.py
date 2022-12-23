import json

import torch
import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from huggingface_hub import cached_download, hf_hub_url
from transformers import SegformerForSemanticSegmentation

from src.config import CFG, Train_CFG
from src.dataset import SegFormerDataset, get_csv_dataset
from src.image_transforms import get_transforms


class SegFormerModule(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.losses_list_train = []
        self.losses_list_valid = []
        self.acc_list_train = []
        self.acc_list_valid = []

        self.id2label = json.load(
            open(
                cached_download(
                    hf_hub_url(
                        "datasets/huggingface/label-files", "ade20k-id2label.json"
                    )
                ),
                "r",
            )
        )
        self.id2label = {int(k): v for k, v in self.id2label.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.id2label = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
        self.label2id = {v: k for k, v in self.id2label.items()}

        print(self.id2label)
        print(self.label2id)

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=8,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        self.lr = Train_CFG.lr

    def forward(self, pixel_values, labels):
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, amsgrad=False)
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=Train_CFG.step, gamma=Train_CFG.gamma
        )
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        images, targets = train_batch
        images, targets = images.type(torch.float32), targets.type(torch.long)

        outputs = self.model(pixel_values=images, labels=targets)
        loss, logits = outputs.loss, outputs.logits
        self.losses_list_train.append(loss.item())
        self.log(
            "train_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_epoch_end(self, outs):
        self.log("train_epoch_loss", np.mean(self.losses_list_train))
        self.losses_list_train = []

    def validation_step(self, val_batch, batch_idx):
        images, targets = val_batch
        images, targets = images.type(torch.float32), targets.type(torch.long)

        outputs = self.model(pixel_values=images, labels=targets)
        loss, logits = outputs.loss, outputs.logits
        self.losses_list_valid.append(loss.item())
        self.log(
            "valid_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_epoch_end(self, outs):
        self.log("valid_epoch_loss", np.mean(self.losses_list_valid))
        self.losses_list_valid = []

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def setup(self, stage=None):

        data_path = f"{CFG.DATA_DIR}/uavid_train/seq1"
        df = get_csv_dataset(data_path)

        train_dataset, valid_dataset = train_test_split(df, test_size=0.1, shuffle=True)
        train_dataset, valid_dataset = train_dataset.reset_index(
            drop=True
        ), valid_dataset.reset_index(drop=True)

        train_dataset = SegFormerDataset(
            data_df=train_dataset, transforms=get_transforms(train=True)
        )
        valid_dataset = SegFormerDataset(
            data_df=valid_dataset, transforms=get_transforms(train=False)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Train_CFG.train_bs,
            shuffle=True,
            num_workers=8,
            # pin_memory=True,
            # collate_fn=self.collate_fn,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=Train_CFG.valid_bs,
            shuffle=False,
            num_workers=8,
            # pin_memory=True,
            # collate_fn=self.collate_fn,
        )

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.tr_loader = train_loader
        self.v_loader = valid_loader

    def train_dataloader(self):
        return self.tr_loader

    def val_dataloader(self):
        return self.v_loader

    def test_dataloader(self):
        return self.v_loader
