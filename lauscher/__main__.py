"""
LAUSCHER – Flexible Auditory Spike Conversion Chain

Reference: https://arxiv.org/abs/1910.07407
"""

import argparse
import logging
from os.path import isfile

from lauscher.audiowaves import FileMonoAudioWave
from lauscher.helpers import CommandLineArguments
from lauscher.transformations.wave2spike import Wave2Spike


def main(input_file: str,
         output_file: str,
         num_channels: int):
    if not isfile(input_file):
        raise IOError(f"Input file '{input_file}' not found.")

    trafo = Wave2Spike(num_channels=num_channels)
    spikes = FileMonoAudioWave(input_file).transform(trafo)
    spikes.export(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=str,
                        help="Path to the input wave file, to be converted to"
                             "a spike train.")
    parser.add_argument("output_file", type=str,
                        help="Path to the output file, spike trains will ber"
                             "written into it.")
    parser.add_argument("--num_channels", type=int, default=700,
                        help="Number of frequency selective channels.")
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help="Number of concurrent jobs used for data "
                             "processing.")
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    global_args = CommandLineArguments()
    global_args.num_concurrent_jobs = args.jobs
    main(args.input_file, args.output_file, args.num_channels)
