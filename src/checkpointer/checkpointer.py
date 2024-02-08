import shutil
from typing import Callable, Union, List
import os
from pathlib import Path
import signal
import sys
from .checkpointing_utils import get_condor_job_ad_settings


class Checkpointer:
    '''
    Class to manage checkpoiniting in different contexts.
    Allows to 
        - store and load checkpoints from remote locations
        - regulary create checkpoints
        - induce checkpoint taking from outside signals
        - reschedule jobs after checkpointing
    '''

    def __init__(
        self,
        # Path or list of Paths of files defining the checkpoint
        # file or list of files defining the checkpoint
        local_checkpoint_file: Path,
        checkpoint_function: Callable,  # function to call to create the checkpoints
        restore_function: Callable,  # function to call to restore the checkpoints
        # how to trasfer the checkpoint files, currently None(default), shared, xrootd, manual and htcondor are supported
        checkpoint_transfer_mode: str = "None",
        # where to store the checkpoint files, if None, the current working directory is used
        checkpoint_transfer_target: Union[str, Path] = None,
        # name of the xrootd server to use in xrootd mode
        xrootd_server_name: str = None,
        # function to call when manual checkpoint_transfer_mode is used
        checkpoint_transfer_callback: Callable = None,
        # kwargs to be used in in checkpoint_transfer
        checkpoint_transfer_callback_kwargs: dict = None,
        checkpoint_every: int = 10,  # how often to create checkpoints
        # function to call before exiting on SIGTERM
        on_SIGTERM_prehook: Callable = None,
        on_SIGTERM_prehook_kwargs: dict = None,  # kwargs to pass to on_SIGTERM_prehook

    ) -> None:
        '''
        Class to manage checkpoiniting in different contexts.
        parameters:
            local_checkpoint_file: Path or list of Paths of files defining the checkpoint in the current enviroment
            checkpoint_function: function to call to create the checkpoints
            restore_function: function to call to restore the checkpoints
            checkpoint_transfer_mode: how to trasfer the checkpoint files, currently None(default), shared, xrootd, manual and htcondor are supported.
            checkpoint_transfer_target: where to store the checkpoint files, if None, the current working directory is used
            xrootd_server_name: name of the xrootd server to use in xrootd mode
            checkpoint_transfer_callback: function to call when manual checkpoint_transfer_mode is used
            checkpoint_transfer_callback_kwargs: kwargs to be used in in checkpoint_transfer
            checkpoint_every: how often to create checkpoints when using the step function
            on_SIGTERM_prehook: function to call before exiting on SIGTERM
            on_SIGTERM_prehook_kwargs: kwargs to pass to on_SIGTERM_prehook
        '''
    

        # if only one checkpoint path is given, convert to list
        assert isinstance(
            local_checkpoint_file, Path
        ), "local_checkpoint_files must be a list of Paths"
        # set parameters
        self.local_checkpoint_file = local_checkpoint_file
        self.checkpoint_function = checkpoint_function
        self.restore_function = restore_function
        self.checkpoint_transfer_mode = checkpoint_transfer_mode
        self.checkpoint_transfer_target = checkpoint_transfer_target
        self.checkpoint_transfer_callback = checkpoint_transfer_callback
        self.checkpoint_transfer_callback_kwargs = checkpoint_transfer_callback_kwargs
        self.checkpoint_every = checkpoint_every
        self.on_SIGTERM_prehook = on_SIGTERM_prehook if on_SIGTERM_prehook else lambda: None
        self.on_SIGTERM_prehook_kwargs = on_SIGTERM_prehook_kwargs if on_SIGTERM_prehook_kwargs else {}
        self.checkpoint_exit_code = 85

        # initialize internal variables
        self.step_counter = 0
        self.checkpoint_value = None

        # register signal handlers
        signal.signal(signal.SIGTERM, self.on_SIGTERM)
        signal.signal(signal.SIGINT, self.on_SIGTERM)

        # setup transfer mode
        assert self.checkpoint_transfer_mode in [
            "None", "shared", "xrootd", "manual", "htcondor"
        ], "checkpoint_transfer_mode must be one of None, shared, xrootd, manual, htcondor"

        if self.checkpoint_transfer_mode == "shared":
            assert isinstance(
                checkpoint_transfer_target, Path
            ), "checkpoint_transfer_target must be a Path in shared mode"

        elif self.checkpoint_transfer_mode == "xrootd":
            assert isinstance(
                checkpoint_transfer_target, str
            ), "checkpoint_transfer_target must be a string in xrootd mode"
            assert local_checkpoint_file.is_absolute(
            ), "local_checkpoint_files must be absolute paths in xrootd mode"
            assert xrootd_server_name is not None, "xrootd_server_name not set"
            from XRootD import client
            from XRootD.client.flags import DirListFlags
            self.DirListFlags = DirListFlags  # need this later for the exists check
            self.xrootd_server_name = xrootd_server_name
            self.xrootd_client = client.FileSystem(xrootd_server_name)

        elif self.checkpoint_transfer_mode == "manual":
            assert self.checkpoint_transfer_callback is not None, "checkpoint_transfer_callback not set"
            assert self.checkpoint_transfer_callback_kwargs is not None, "checkpoint_transfer_callback_kwargs not set"

        elif self.checkpoint_transfer_mode == "htcondor":
            self.checkpoint_transfer_target = Path(
                get_condor_job_ad_settings("TransferCheckpoint")
            )
            assert self.checkpoint_transfer_target != "", "transfer_checkpoint_files not set in condor job ad"
            self.checkpoint_exit_code = int(  # Warning! This overwrites the checkpoint_exit_code set above
                get_condor_job_ad_settings("CheckpointExitCode")
            )

    def on_SIGTERM(self, signalNumber, frame):
        '''
        Fuction to call when SIGTERM is received. Calls on_SIGTERM_prehook and exits with checkpoint_exit_code.
        Arguments are only used to match the signal handler signature.
        '''
        self.on_SIGTERM_prehook(**self.on_SIGTERM_prehook_kwargs)
        self.checkpoint()
        self.transfer_checkpoint_files()
        self.clean_up_local_checkpoint_files()
        sys.exit(self.checkpoint_exit_code)

    def clean_up_local_checkpoint_files(self):
        '''
        Removes local checkpoint files if transfer mode is not None.
        '''
        if self.checkpoint_transfer_mode == "None" or self.checkpoint_transfer_mode == "htcondor":
            return
        self.local_checkpoint_file.unlink()

    def checkpoint(self, value=None):
        '''
        Function to create a checkpoint. It calls checkpoint_function with local_checkpoint_files and value as arguments.
        The checkpoint_function should store the checkpoint in the files given in local_checkpoint_files.
        The checkpoint_function receives the local_checkpoint_file and the value as arguments.
        If value is None, the last checkpoint_value is used.
        '''
        if value is None:
            value = self.checkpoint_value
        self.checkpoint_function(self.local_checkpoint_file, value)
        self.checkpoint_value = value

    def restore(self, default):
        '''
        Function to restore a checkpoint. Calls restore_function with local_checkpoint_file as argument.
        If no checkpoint exists, default is returned.
        '''
        self.get_checkpoint()
        if self.restore_function and self.local_checkpoint_file.exists():
            return self.restore_function(self.local_checkpoint_file)
        return default

    def transfer_checkpoint_files(self):
        '''
        Function to transfer checkpoint files to a remote location. Used in shared, xrootd and manual mode.
        In manual mode, the checkpoint_transfer_callback is called with local_checkpoint_files, checkpoint_transfer_target and checkpoint_transfer_callback_kwargs as arguments.
        '''
        if self.checkpoint_transfer_mode == "None" or not self.local_checkpoint_file.exists():
            return

        if self.checkpoint_transfer_mode == "shared":
            shutil.copy(self.local_checkpoint_file, self.checkpoint_transfer_target)

        elif self.checkpoint_transfer_mode == "xrootd":
            status, _ = self.xrootd_client.copy(
                'file://' + str(self.local_checkpoint_file),
                self.xrootd_server_name + self.checkpoint_transfer_target, force=True
            )
            if not status.ok:
                print(status.message)

        elif self.checkpoint_transfer_mode == "manual":
            self.checkpoint_transfer_callback(
                self.local_checkpoint_file,
                self.checkpoint_transfer_target,
                **self.checkpoint_transfer_callback_kwargs)

        elif self.checkpoint_transfer_mode == "htcondor":
            pass

    @property
    def checkpoint_exists(self):
        '''
        Property to check if a previous checkpoint exists. 
        Without transfer, this is just a check if the local_checkpoint_files exist.
        In shared and xrootd mode, this is a check if the checkpoint_transfer_target exists.
        '''
        if self.checkpoint_transfer_mode == "None":
            return self.local_checkpoint_file.exists()

        elif self.checkpoint_transfer_mode == "shared":
            return self.checkpoint_transfer_target.exists()

        elif self.checkpoint_transfer_mode == "xrootd":

            status, listing = self.xrootd_client.stat(
                self.checkpoint_transfer_target, self.DirListFlags.STAT
            )
            return status.ok

    def get_checkpoint(self):
        '''
        Function to get the checkpoint files from a remote location. Used in shared and xrootd mode.
        '''
        # TODO: implement manual mode

        if self.checkpoint_exists:
            if self.checkpoint_transfer_mode == "shared":
                os.system(
                    "cp {} {}".format(
                        self.checkpoint_transfer_target._str,
                        str(self.local_checkpoint_file)
                    )
                )

            elif self.checkpoint_transfer_mode == "xrootd":
                status, _ = self.xrootd_client.copy(
                    self.xrootd_server_name +
                    self.checkpoint_transfer_target,
                    str(self.local_checkpoint_file),
                )
                if not status.ok:
                    print(status.message)
            elif self.checkpoint_transfer_mode == "manual":
                self.checkpoint_transfer_callback(
                    self.checkpoint_transfer_target,
                    self.local_checkpoint_file,
                    **self.checkpoint_transfer_callback_kwargs
                )

    def step(self, value):
        '''
        Function to call to create a checkpoint every checkpoint_every steps.
        Used for compatiblity with pytorch-lightning, tensorflow and other frameworks.
        '''
        self.checkpoint_value = value
        if self.step_counter % self.checkpoint_every == 0:
            self.checkpoint(value)
            self.transfer_checkpoint_files()
        self.step_counter += 1
