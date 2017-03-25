# Characterizing RNA Pseudouridylation by Convolutional Neural Networks--PULSE model
## Directories
1. The 'model' directory includes PULSE models and theirs weights and architecture.

2. The 'data' directory contains the analysis files we used in the article.
## Prerequisites
* Python 2.7
* Keras 1.0.5
* Numpy 1.12.0
* Theano 0.9
* CUDA 8.0
## Usage
1. Change to the 'model' directory
2. Use the following command to run PULSE:
'''
python pulse.py -s [human(h) or mouse(m)] -i input.fa -o out_path
'''
