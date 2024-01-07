from pytorch_lightning.callbacks import Callback
from checkpointer.checkpointer import Checkpointer

class LightningCheckpointerCallback(Callback):
    def __init__(self, **checkpointer_kwargs) -> None:
        super().__init__()
        self.checkpointer = Checkpointer(
            checkpoint_function = lambda path, trainer: trainer.save_checkpoint(path),
            restore_function = lambda path: path,
            **checkpointer_kwargs
        )

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self.checkpointer.transfer_checkpoint_files()

    def restore(self):
        return self.checkpointer.restore(None)
        
    def on_train_epoch_end(self, trainer, pl_module):
        self.checkpointer.step(trainer)