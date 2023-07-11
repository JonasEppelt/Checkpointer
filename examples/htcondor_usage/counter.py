import time
from pathlib import Path

from checkpointer.checkpointer import Checkpointer


checkpointer = Checkpointer(
    local_checkpoint_files=Path("checkpoint.txt"),
    restore_function=lambda path: int(path.read_text()),
    checkpoint_function=lambda path, value: path.write_text(str(value)),
    checkpoint_every=100,
    batch_system_mode="htcondor",
)
start_value = checkpointer.restore(0)
print("starting at: ", start_value)
for i in range(start_value, 10_000):
    print(i)
    checkpointer.step(i)
    time.sleep(0.1)

print("finished counting")
