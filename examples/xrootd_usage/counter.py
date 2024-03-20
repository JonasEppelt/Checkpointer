import time
from pathlib import Path

from checkpointer.checkpointer import Checkpointer


checkpointer = Checkpointer(
    local_checkpoint_file=Path("checkpoint.txt").absolute(),
    restore_function=lambda path: int(path.read_text()),
    checkpoint_function=lambda path, value: path.write_text(str(value)),
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
