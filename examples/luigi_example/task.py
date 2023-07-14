import luigi
from checkpointer.checkpointer import Checkpointer
from pathlib import Path
import time
import numpy as np


class Counter(luigi.Task):

    def requires(self):
        return None

    def output(self):
        return luigi.LocalTarget("/work/jeppelt/result.txt")

    def run(self):
        checkpointer = Checkpointer(
            local_checkpoint_files=Path("checkpoint.txt"),
            restore_function=lambda paths: int(paths[0].read_text()),
            checkpoint_function=lambda paths, value: paths[0].write_text(
                str(value)),
            checkpoint_every=10,
            checkpoint_transfer_mode="shared",
            checkpoint_transfer_target=Path("/work/jeppelt/checkpoint.txt"),
        )
        start_value = checkpointer.restore(0)
        print("starting at: ", start_value)
        for i in range(start_value, 10_000):
            print(i)
            checkpointer.step(i)
            time.sleep(1)
            if np.random.rand() < 0.1:
                raise RuntimeError("random error")

        print("finished counting")
        # write result
        self.output().makedirs()
        with self.output().open("w") as f:
            f.write(i)


class Counter2(Counter):

    def output(self):
        return luigi.LocalTarget("/work/jeppelt/result.txt")

    def requires(self):
        return Counter()


if __name__ == "__main__":
    luigi.build([Counter2()], local_scheduler=False, workers=1,
                scheduler_host="bms1.etp.kit.edu", scheduler_port=8022)
