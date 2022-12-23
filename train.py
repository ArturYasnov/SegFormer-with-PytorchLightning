import pytorch_lightning as pl
import torch

from src.config import CFG, Train_CFG
from src.lightning_module import SegFormerModule


if __name__ == "__main__":
    torch.cuda.empty_cache()
    model = SegFormerModule()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=Train_CFG.epocs,
        precision=32,
        gradient_clip_val=1e-1,
        fast_dev_run=False,
        profiler=None,
        accumulate_grad_batches=Train_CFG.accumulate_bs,
        callbacks=None,
    )

    trainer.fit(model)

    torch.save(
        model.model.state_dict(), f"{CFG.MODELS_DIR}/{Train_CFG.experiment_name}.pt"
    )
