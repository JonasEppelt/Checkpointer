from keras.callbacks import BackupAndRestore
from checkpointer.checkpointer import Checkpointer
from pathlib import Path
import os
import tarfile

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir), recursive=True)

def extract_tar(tar_filename, extract_folder):
    with tarfile.open(tar_filename, 'r') as tar:
        tar.extractall(extract_folder)


class KerasCheckpointerCallback(BackupAndRestore):
    '''
    A keras callback to interface the checkpointer with the BackupAndRestore callback. 
    Since keras creates a whole dir structure, it will zip it before transfering.
    The checkpointer's `step` function is executed after the BackupAndRestore's `on_epoch_end` function. 
    The creation frequency of checkpoints must be controlled via the BackupAndRestore's `save_freq` parameter.
    The checkpointers `checkpoint_every` parameter will only determine how often the zip archive will be created.
    Checkpoints are restored before the training begins.
    With the `**checkpointer_kwargs` you can pass configurations like the checkpoint transfer mode to the checkpointer.
    '''
    def __init__(self, local_checkpoint_file, save_freq="epoch", **checkpointer_kwargs) -> None:
        super().__init__(
            backup_dir = local_checkpoint_file,
            save_freq = save_freq,
            delete_checkpoint = True,
        )
        self.zip_file = f"{local_checkpoint_file}/checkpoint.zip"
        self.local_parent_dir = os.path.dirname(local_checkpoint_file)
        self.checkpointer = Checkpointer(
            local_checkpoint_file=Path(self.zip_file),
            checkpoint_function = lambda path, model: make_tarfile(self.zip_file, local_checkpoint_file) , # needing to zip, since keras creates a whole dir structure
            restore_function = lambda path: extract_tar(self.zip_file, self.local_parent_dir), # needing to unzip, since keras creates a whole dir structure
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