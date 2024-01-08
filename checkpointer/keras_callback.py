from keras.callbacks import BackupAndRestore
from checkpointer.checkpointer import Checkpointer
from pathlib import Path
import shutil

class KerasCheckpointerCallback(BackupAndRestore):
    def __init__(self, local_checkpoint_file, **checkpointer_kwargs) -> None:
        super().__init__(
            backup_dir = local_checkpoint_file,
            save_freq = "epoch",
            delete_checkpoint = True,
        )
        self.zip_file = f"{local_checkpoint_file}/checkpoint.zip"
        self.checkpointer = Checkpointer(
            local_checkpoint_file=Path(self.zip_file),
            checkpoint_function = lambda path, model: shutil.make_archive(f"{local_checkpoint_file}/checkpoint", "zip", local_checkpoint_file) , # needing to zip, since keras creates a whole dir structure
            restore_function = lambda path: shutil.unpack_archive(self.zip_file, local_checkpoint_file), # needing to unzip, since keras creates a whole dir structure
            **checkpointer_kwargs
        )

    def on_train_begin(self, logs=None):
        self.checkpointer.restore(self.backup_dir)
        super().on_train_begin(logs)
    
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.checkpointer.step(self.backup_dir)
    
    def backup(self):
        super()._backup(self._current_epoch, self._batches_count)