from pytorch_lightning.callbacks import Callback
from checkpointer.checkpointer import Checkpointer

class LightningCheckpointerCallback(Callback):
    '''
    A callback to interface the checkpointer with the pytorch lightning trainer.
    It uses the trainers `save_checkpoint`´to create the checkpoint and handles the transfer of checkpoint upon their creation.
    Besides the predefined `restore_function` and `checkpoint_function` the callback can be configured just like the checkpointer.
    Since the checkpoint function is also called upon interuption, the training can be resumed from the last batch, that was passed.
    '''
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