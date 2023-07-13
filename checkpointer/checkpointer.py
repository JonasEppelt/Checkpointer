from typing import Callable, Union
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
        local_checkpoint_files: Path = None,
        restore_function: Callable = None,  # function to call to restore the checkpoints
        # checkpoint every n steps when used in the context of a training loop
        checkpoint_function: Callable = None,
        checkpoint_exit_code: int = None,  # exit code to use when checkpointing
        # signal to listen to for induced checkpointing
        induce_checkpoint_signal: int = None,
        # whether to use a batchsystem, currently only HTCondor is supported
        batch_system_mode: str = "None",
        # how to transfer the checkpoint files, currently None(default), shared, xrootd, manual, and htcondor are supported
        checkpoint_transfer_mode: str = "None",
        # where to store the checkpoint files, if None, the current working directory is used
        checkpoint_transfer_target: Union[str, Path] = None,
        # function to call when manual checkpoint_transfer_mode is used
        checkpoint_transfer_callback: Callable = None,
        # kwargs to be used in in checkpoint_transfer
        checkpoint_transfer_callback_kwargs: dict = None,
        # counter to use in training loops, can be set if checkpointing is resumed
        checkpoint_every: int = 10,
        # function to call to create the checkpoints
        step_counter: int = 0,
        # function to call before exiting on SIGTERM
        on_SIGTERM_prehook: Callable = None,
        on_SIGTERM_prehook_kwargs: dict = None,  # kwargs to pass to on_SIGTERM_prehook

    ) -> None:
        # if paths are given as strings, convert them to Path objects
        if isinstance(local_checkpoint_files, str):
            local_checkpoint_files = Path(local_checkpoint_files)
        if isinstance(checkpoint_transfer_target, str):
            checkpoint_transfer_target = Path(checkpoint_transfer_target)
        # set global default values
        self.batch_system_mode = batch_system_mode
        self.induce_checkpoint_signal = induce_checkpoint_signal
        self.step_counter = step_counter
        self.checkpoint_every = checkpoint_every
        self.checkpoint_value = None

        # set up checkpointing
        self.checkpoint_function = checkpoint_function
        self.restore_function = restore_function

        # set file transfer
        self.checkpoint_transfer_mode = checkpoint_transfer_mode
        self.checkpoint_transfer_target = checkpoint_transfer_target
        self.checkpoint_transfer_callback = checkpoint_transfer_callback
        self.checkpoint_transfer_callback_kwargs = checkpoint_transfer_callback_kwargs

        # set up prehooks
        self.on_SIGTERM_prehook = on_SIGTERM_prehook if on_SIGTERM_prehook else lambda: None
        self.on_SIGTERM_prehook_kwargs = on_SIGTERM_prehook_kwargs if on_SIGTERM_prehook_kwargs else {}

        # set up batchsystem mode
        assert batch_system_mode in [
            "None", "HTCondor"], "batch_system_mode must be one of None, HTCondor"
        if batch_system_mode == "None":
            self.local_checkpoint_files = local_checkpoint_files
            self.checkpoint_exit_code = checkpoint_exit_code

        elif batch_system_mode == "HTCondor":
            self.set_condor_default_values()

        # register signal handlers
        signal.signal(signal.SIGTERM, self.on_SIGTERM)
        if self.induce_checkpoint_signal:
            signal.signal(
                self.induce_checkpoint_signal,
                self.on_InducedCheckpointSignal,
            )

        # check correct settings for checkpoint_transfer_mode
        assert self.checkpoint_transfer_mode in [
            "None", "shared", "xrootd", "manual", "htcondor"], "checkpoint_transfer_mode must be one of None, shared, xrootd, manual, htcondor"
        if self.checkpoint_transfer_mode != "None":
            assert self.checkpoint_transfer_target is not None, "checkpoint_transfer_target not set"
        if self.checkpoint_transfer_mode == "shared":
            assert isinstance(
                self.checkpoint_transfer_target,
                Path,), "checkpoint_transfer_target must be a Path in shared checkpoint_transfer_mode"
        elif self.checkpoint_transfer_mode == "xrootd":
            assert self.checkpoint_transfer_target is not str, "checkpoint_transfer_target must be a string in xrootd mode"
            assert self.checkpoint_transfer_callback_kwargs is not None, "checkpoint_transfer_callback_kwargs not set"
            assert "xrootd_server_name" in self.checkpoint_transfer_callback_kwargs.keys(
            ), "xrootd_server_name not set in checkpoint_transfer_callback_kwargs"

    def set_condor_default_values(self):
        self.local_checkpoint_files = Path(get_condor_job_ad_settings(
            "TransferCheckpoint"))
        self.induce_checkpoint_signal = get_condor_job_ad_settings(
            "+SuccessCheckpointExitSignal")
        self.checkpoint_exit_code = get_condor_job_ad_settings(
            "checkpoint_exit_code")

    def write_pid(self):
        # write pid to file
        with open(self.pid_file, "w") as file:
            file.write(str(self.pid))

    def on_SIGTERM(self, signalNumber, frame):
        print("on_SIGTERM, Received: ", signalNumber)
        self.on_SIGTERM_prehook(**self.on_SIGTERM_prehook_kwargs)
        self.checkpoint()
        sys.exit(self.checkpoint_exit_code)

    def on_InducedCheckpointSignal(self, signalNumber, frame):
        print("on_InducedCheckpointSignal Received: ", signalNumber)
        self.checkpoint()

    def checkpoint(self, value=None):
        if value is None:
            value = self.checkpoint_value
        self.checkpoint_function(self.local_checkpoint_files, value)

    def restore(self, default=None):
        self.get_checkpoint()
        if self.restore_function and self.local_checkpoint_files.exists():
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
