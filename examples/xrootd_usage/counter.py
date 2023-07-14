import time
from pathlib import Path

from checkpointer.checkpointer import Checkpointer


checkpointer = Checkpointer(
    local_checkpoint_files=Path("checkpoint.txt").absolute(),
    restore_function=lambda paths: int(paths[0].read_text()),
    checkpoint_function=lambda paths, value: paths[0].write_text(str(value)),
    checkpoint_every=100,
    checkpoint_transfer_mode="xrootd",
    checkpoint_transfer_target="/pnfs/gridka.de/belle/disk-only/LOCAL/user/jeppelt/checkpoint.txt",
    xrootd_server_name="root://dcachexrootd-kit.gridka.de:1094",
)
start_value = checkpointer.restore(0)
print("starting at: ", start_value)
for i in range(start_value, 10_000):
    print(i)
    checkpointer.step(i)
    time.sleep(1)

print("finished counting")
