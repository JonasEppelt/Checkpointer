from typing import Callable, Union, List
import os
from pathlib import Path
import signal
import sys
from checkpointer.checkpointing_utils import get_condor_job_ad_settings
# from XRootD import client
# from XRootD.client.flags import DirListFlags, OpenFlags, MkDirFlags


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
        local_checkpoint_files: Union[Path, List[Path]],
        checkpoint_function: Callable,  # function to call to create the checkpoints
        restore_function: Callable,  # function to call to restore the checkpoints
        # batch system to use, currently None(default) and HTCondor are supported
        job_reschedule_mode: str = "None",
        # how to trasfer the checkpoint files, currently None(default), shared, xrootd, manual and htcondor are supported
        checkpoint_transfer_mode: str = "None",
        # where to store the checkpoint files, if None, the current working directory is used
        checkpoint_transfer_target: Union[str, Path] = None,
        # function to call when manual checkpoint_transfer_mode is used
        checkpoint_transfer_callback: Callable = None,
        # kwargs to be used in in checkpoint_transfer
        checkpoint_transfer_callback_kwargs: dict = None,
        checkpoint_every: int = 10,  # how often to create checkpoints
        # function to call before exiting on SIGTERM
        on_SIGTERM_prehook: Callable = None,
        on_SIGTERM_prehook_kwargs: dict = None,  # kwargs to pass to on_SIGTERM_prehook

    ) -> None:

        # if only one checkpoint path is given, convert to list
        if isinstance(local_checkpoint_files, Path):
            local_checkpoint_files = [local_checkpoint_files]
        else:
            assert all(
                [isinstance(
                    path, Path) for path in local_checkpoint_files]), "local_checkpoint_files must be a list of Paths"

        # set parameters
        self.local_checkpoint_files = local_checkpoint_files
        self.checkpoint_function = checkpoint_function
        self.restore_function = restore_function
        self.job_reschedule_mode = job_reschedule_mode
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

        # setup transfer mode
        assert self.checkpoint_transfer_mode in [
            "None", "shared", "xrootd", "manual", "htcondor"
        ], "checkpoint_transfer_mode must be one of None, shared, xrootd, manual, htcondor"
        if self.checkpoint_transfer_mode == "shared":
            if isinstance(self.checkpoint_transfer_target, str):
                self.checkpoint_transfer_target = Path(
                    self.checkpoint_transfer_target)
            assert isinstance(
                self.checkpoint_transfer_target,
                Path), "checkpoint_transfer_target must be a Path in shared mode"
        elif self.checkpoint_transfer_mode == "xrootd":
            assert isinstance(
                self.checkpoint_transfer_target, str
            ), "checkpoint_transfer_target must be a string in xrootd mode"
            assert self.checkpoint_transfer_callback_kwargs is not None, "checkpoint_transfer_callback_kwargs not set"
            assert "xrootd_server_name" in self.checkpoint_transfer_callback_kwargs.keys(
            ), "xrootd_server_name not set in checkpoint_transfer_callback_kwargs"
        elif self.checkpoint_transfer_mode == "manual":
            assert self.checkpoint_transfer_callback is not None, "checkpoint_transfer_callback not set"
            assert self.checkpoint_transfer_callback_kwargs is not None, "checkpoint_transfer_callback_kwargs not set"
        elif self.checkpoint_transfer_mode == "htcondor":
            self.checkpoint_transfer_target = Path(
                get_condor_job_ad_settings("TransferCheckpoint"))
            print("transfer_checkpoint_files: ",
                  self.checkpoint_transfer_target)
            assert self.checkpoint_transfer_target != "", "transfer_checkpoint_files not set in condor job ad"

        # setup rescheduling mode
        if self.job_reschedule_mode == "htcondor":
            self.checkpoint_exit_code = get_condor_job_ad_settings(
                "SuccessCheckpointExitCode")
            assert self.checkpoint_exit_code != "", "checkpoint_exit_code not set in condor job ad"

    def on_SIGTERM(self, signalNumber, frame):
        print("on_SIGTERM, Received: ", signalNumber)
        self.on_SIGTERM_prehook(**self.on_SIGTERM_prehook_kwargs)
        self.checkpoint()
        sys.exit(self.checkpoint_exit_code)

    def checkpoint(self, value=None):
        if value is None:
            value = self.checkpoint_value
        self.checkpoint_function(self.local_checkpoint_files, value)

    def restore(self, default=None):
        self.get_checkpoint()
        if self.restore_function and all(file.exist() for file in self.local_checkpoint_files):
            return self.restore_function(self.local_checkpoint_files)
        return default

    def transfer_checkpoint_files(self):
        if self.checkpoint_transfer_mode == "None" or not self.local_checkpoint_files.exists():
            # nothing to do
            return
        assert self.checkpoint_storage_location is not None, "checkpoint_storage_location not set"
        if self.checkpoint_transfer_mode == "shared":
            # use cp to copy checkpoint on local system
            os.system(
                "cp {} {}".format(
                    self.local_checkpoint_files,
                    self.checkpoint_transfer_target,))
        elif self.checkpoint_transfer_mode == "xrootd":
            xrootd_server_name = self.checkpoint_transfer_callback_kwargs["xrootd_server_name"]
            xrootd_client = client.FileSystem(xrootd_server_name)
            status = xrootd_client.copy(
                'file://' + self.local_checkpoint_files,
                xrootd_server_name + self.checkpoint_transfer_target,)
            if not status.ok:
                print(status.message)
        elif self.checkpoint_transfer_mode == "manual":
            self.checkpoint_transfer_callback(
                **self.checkpoint_transfer_callback_kwargs)
        elif self.checkpoint_transfer_mode == "htcondor":
            pass

    @property
    def checkpoint_exists(self):
        if self.checkpoint_transfer_mode == "None":
            return self.local_checkpoint_files.exists()
        elif self.checkpoint_transfer_mode == "shared":
            return self.checkpoint_transfer_target.exists()
        elif self.checkpoint_transfer_mode == "xrootd":
            xrootd_server_name = self.checkpoint_transfer_callback_kwargs["xrootd_server_name"]
            xrootd_client = client.FileSystem(xrootd_server_name)
            status, listing = self.client.stat(
                self.checkpoint_transfer_target, DirListFlags.STAT)
            if not status.ok:
                return False
            else:
                return True

    def get_checkpoint(self):
        if self.checkpoint_exists:
            if self.checkpoint_transfer_mode == "shared":
                os.system("cp {} {}".format(
                    self.checkpoint_transfer_target, self.local_checkpoint_files))
            elif self.checkpoint_transfer_mode == "xrootd":
                xrootd_server_name = self.checkpoint_transfer_callback_kwargs["xrootd_server_name"]
                xrootd_client = client.FileSystem(xrootd_server_name)
                status = xrootd_client.copy(
                    xrootd_server_name + self.checkpoint_transfer_target,
                    self.local_checkpoint_files,)
                if not status.ok:
                    print(status.message)

    def step(self, value):
        self.checkpoint_value = value
        if self.step_counter % self.checkpoint_every == 0:
            self.checkpoint(value)
            self.transfer_checkpoint_files()
        self.step_counter += 1
