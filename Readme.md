<p align="center">
  <img src=".artwork/logo.png" width="80%" alt="LAUSCHER Logo"/>
</p>

# LAUSCHER – Flexible Auditory Spike Conversion Chain

If you find this package useful for your scientific work, please consider citing our paper 
https://ieeexplore.ieee.org/document/9311226
```
@article{cramer_heidelberg_2020,
	title = {The {Heidelberg} {Spiking} {Data} {Sets} for the {Systematic} {Evaluation} of {Spiking} {Neural} {Networks}},
	issn = {2162-2388},
	doi = {10.1109/TNNLS.2020.3044364},
	journal = {IEEE Transactions on Neural Networks and Learning Systems},
	author = {Cramer, B. and Stradmann, Y. and Schemmel, J. and Zenke, F.},
	year = {2020},
	pages = {1--14}
}
```

## Installation
*Lauscher* uses the [SoundFile](https://pysoundfile.readthedocs.io/en/latest/) package for parsing audio data.
On Linux, this library requires you to install `libsndfile` via your distribution's package manager, e.g. via `apt install libsndfile1`.

We recommend setting up a [virtualenv](https://github.com/pypa/virtualenv) (you'll need `apt install python3-venv` or similar) for your *lauscher* project:
```shell
git clone https://github.com/electronicvisions/lauscher.git
cd lauscher
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
Assuming you are working in a private copy of *lauscher* that has been installed as noted above, you can continue by calculating a first spiketrain from one of the included examples:
```shell
python -m lauscher test/resources/spoken_digit.flac my_first_spiketrain.npz --num_channels 70 --verbose
```
Note that due to the complexity of the implemented model, even short audio files might take multiple minutes to convert.
Memory consumption might be huge for long audio files.

## License
```
LAUSCHER - Flexbile Auditory Spike Conversion Chain
Copyright (C) 2020 Benjamin Cramer
                   Yannik Stradmann
                   Koshika Yadava

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
```
