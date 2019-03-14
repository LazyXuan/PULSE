# PseudoUridyLation Sites Estimator -- PULSE

## Directories
1. The 'model' directory includes PULSE models and theirs weights and architecture.
>* 'cnn_structure.json': saved the network structure of the PULSE model
>* 'hPULSE_weights.h5': is the weight of hPULSE model
>* 'mPULSE_weights.h5': is the weight of mPULSE model
>* 'pulse.py': is the predictor which can be used to predict new pseudouridine sites
>* 'utils.py': include some useful functions which were imported by 'pulse.py'

2. The 'data' directory contains the analysis files we used in the article.
>* 'GO_enrichment_analysis': includes the GO enrichment results which were showed in our manuscript
>* 'SNVs': includes the information of single nucleotide variants
>* 'mRNA_stability': includes the data we calculated about RNA stability
>* 'motifs': includes the motifs we called from our models
>* 'sequences': the sequence samples we used to train, test and validate our models
>* 'translation': includes the ribo-seq and predicted psi values

3. The 'retrained' directory contains the data and results we used to retrain our models.
>* 'prediction_results': the prediction results of retrained hPULSE and mPULSE on datasets with different negative-to-positive folds (1, 5, 10 and 20 folds)
>* 'regenerate_sequence_samples': the sequence samples in different negative-to-positive folds (1, 5, 10 and 20 folds) and their corresponded labels
>* 'retrained_cnn_models': the retrained model scripts
>* 'retrained_cnn_weights': weights of the retrained models
>* 'validation_results': prediction results iRNAPseU and PPUS on high quality independent datasets

## Prerequisites
* Python 2.7
* Keras 1.0.5
* Numpy 1.12.0
* Theano 0.9
* CUDA 8.0

## Usage
To run this model you don't need to install the scripts, just download the whole PULSE directory and then:
1. Change to the 'model' directory
2. Use the following command to run PULSE:
```
python pulse.py -s [human(h) or mouse(m)] -i input.fa -o out_path
```
>* -s: --species, choose a species, human (h) or mouse (m)
>* -i: the input fasta files you want to predict
>* -o: output file include the prediction result

## Notes
1. The input sequence should be in one line fasta format.
2. The 'data' directory contains the results of the paper

If you have any questions, please feel free to contact me <br />
E-mail: he-x15@mails.tsinghua.edu.cn
