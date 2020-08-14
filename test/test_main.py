import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from lauscher.__main__ import main


class TestLauscher(unittest.TestCase):
    VALID_INPUT_FILE = Path(__file__).parent.joinpath("resources",
                                                      "spoken_digit.flac")

    def test_transformation(self):
        with TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir).joinpath("new_subfolder/out.npz")
            main(self.VALID_INPUT_FILE, output_file, 70)

            self.assertGreater(os.path.getsize(output_file), 0,
                               "Output file has not been created or is empty!")

    def test_exceptions(self):
        non_existing = Path(__file__).parent.joinpath("doesnotexist.flac")
        with self.assertRaises(IOError):
            main(non_existing, "something.npz", 70)


if __name__ == "__main__":
    unittest.main()
