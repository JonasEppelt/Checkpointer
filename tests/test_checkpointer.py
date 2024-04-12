import unittest
from pathlib import Path
from checkpointer.checkpointer import Checkpointer


class TestCheckpointer(unittest.TestCase):
    def setUp(self):
        self.checkpointer = Checkpointer(
            local_checkpoint_file=Path("checkpoint.txt"),
            restore_function=lambda path: int(path.read_text()),
            checkpoint_function=lambda path, value: path.write_text(str(value)),
            checkpoint_every=100,
        )  

    def test_loop(self):
        start_value = self.checkpointer.restore(0)
        print("starting at: ", start_value)
        for i in range(start_value, 10_000):
            self.checkpointer.step(i)
        
        load_checkpoint = self.checkpointer.restore(0)
        self.assertEqual(load_checkpoint, 9900)

