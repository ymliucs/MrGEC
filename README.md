<div align="center">

# Towards Better Utilization of Multi-Reference Training Data for Chinese Grammatical Error Correction
__Yumeng Liu__, Zhenghua Li<sup title="Corresponding author" style="font-size:10px">✉️</sup>, Haochen Jiang, Bo Zhang, Chen Li, Ji Zhang

</div>

## Abstract
This repo contains the code for our ACL 2024 Findings paper: Towards Better Utilization of Multi-Reference Training Data for Chinese Grammatical Error Correction.

## Set up
1. Prepare the conda environment for MrGEC
```shell
conda create -n mrgec python==3.10.10
conda activate mrgec
pip install -r requirements.txt
python -m spacy download en
```

2. Prepare the conda environment for the evaluation tool ChERRANT
```shell
conda create -n cherrant python==3.8
conda activate cherrant
pip install -r utils/ChERRANT/requirements.txt
```

3. Download the pre-trained models
```shell
python utils/download.py --repo_id HillZhang/pseudo_native_bart_CGEC
python utils/download.py --repo_id fnlp/bart-large-chinese
```

Before running, you are required to preprocess each instance into the format of 
```txt
S   [src]
T   [tgt1]
T   [tgt2]
T   [tgt3]

S   [src]
T   [tgt1]
T   [tgt2]
```
Where `[src]` and `[tgt]` are the source and target sentences, respectively.
A `\t` is used to separate the prefix `S` or `T` and the sentence.
Each instance is separated by a blank line.

## Handle data leakage
We find the FCGEC-Train and NaSGEC-Exam/NaCGEC have a severe **data leakage** problem. The code in *utils/handle_data_leakage_tool* can handle all the Chinese GEC datasets which have data leakage problem.

All the datasets need to be processed to the follow format:
```txt
[idx] [src] [tgt1] [tgt2] ... 
```

Where `[idx]` is the index number of an instance which starts with 1, and the sentences are separate by `\t`. 

Usage example:
```
python handle_leakage.py --data_dir data/ns_original --out_dir data/ns_leakage_processed --train_file FCGEC_train_filtered.para --extract_test_files nasgec.exam.para,nacgec.all.para  --frozen_test_files fcgec.dev.para,fcgec.test.para

python handle_leakage.py --data_dir data/ns_original --out_dir data/ns_leakage_processed --train_file FCGEC_train_filtered.para --extract_test_files nasgec.exam.para

python handle_leakage.py --data_dir data/ns_original --out_dir data/ns_leakage_processed --train_file FCGEC_train_filtered.para --frozen_test_files nasgec.exam.para
```


## Download data

You can download all the data we use [here](https://drive.google.com/file/d/1rUjCLu7m4lYvlozOrJ83AzTdaVW1i5w2/view?usp=sharing).

## Run
You can see all the commands for running our experiments in the *bash* folder, and the hyperparameters can be set in the *configs* folder.

Examples:
```shell
bash bash/run_lang8_cat.sh
bash bash/run_lang8_avgl_minl.sh
bash bash/run_fcgec_cat.sh
bash bash/run_fcgec_avgl_minl.sh
```

You can download and check all the logs of our experiments [here](https://drive.google.com/file/d/1qqcL0-eQgTUjr9RlEKTxLIhAgJ749L6j/view?usp=sharing).

## Acknowledgements
1. This repository is completely based on [SuPar](https://github.com/yzhangcs/parser).
2. We use the [ChERRANT](https://github.com/HillZhang1999/MuCGEC/tree/main/scorers/ChERRANT) for all the evaluation.

## Citation
If you find this repo helpful, please cite the following paper:
```bib
@inproceedings{liu-etal-2024-towards-better,
    title = "{Towards Better Utilization of Multi-Reference Training Data for {C}hinese Grammatical Error Correction}",
    author = "Liu, Yumeng  and
      Li, Zhenghua  and
      Jiang, HaoChen  and
      Zhang, Bo  and
      Li, Chen  and
      Zhang, Ji",
    booktitle = "Findings of ACL",
    year = "2024",
    address = "Bangkok, Thailand",
    url = "https://aclanthology.org/2024.findings-acl.180/",
    pages = "3044--3052",
}
```
