The official implementation of paper "A Phrase-level Attention Enhanced CRF for Keyphrase Extraction".

# About our PAE-CRF
This is the framework of our paecrf. 

![framework of paecrf](./pic/framework.png)

This is the results between PAE-CRF and other baselines.

![main result](./pic/result1.png)
![number](./pic/result2.png)
![ablated result](./pic/result3.png)
![example](./pic/result4.png)

# Code
This code is mainly adapted from [bert4torch](https://github.com/Tongjilibo/bert4torch). Thanks for his work.

# Quick Start
The whole process includes the following steps: Preprocess, Training, Evaluation.

The original datasets is provided by [kenchan](https://github.com/kenchan0226/keyphrase-generation-rl).
The original and labeled datasets and our PAE-CRF's results can be download from [here](https://drive.google.com/file/d/1g6Rkyk0Jbcd0Vv2kJwqU5PaiP-VP_str/view?usp=sharing).
Please read the `readme.md` under `/datasets` to get more details.

The pre-trained models, consisting of bert-base-uncased to train word embedding and all-mpnet-base-v1 to rank keyphrases, can be download from [here](https://drive.google.com/file/d/1Rwb3y0sniiahZ6Sb8ZhlFWMQnuMWwbiq/view?usp=sharing).
Please download and replace them under `/pretrain_models`.

The trained parameters of our PAE-CRF can be download from [here](https://drive.google.com/file/d/1aUglIQRaaFXk-9JKE0RTpEeEmMDVAKc2/view?usp=sharing).
Please download and replace it under `/checkpoints`.

# Preprocess
We label the datasets with BIOES scheme.

When two keyphrases overlap, the latter keyphrase is not labeled to ensure that each word has unique word-level label.
For example, given an input text `Abductive network committees for improved classification of medical data` and the corresponding keyphrases `abductive networks` and `network committee`, following the previous labeling method, the first three words are labeled as $\{B_{CW}, B_{CW}, E_{CW}\}$. 
Since the word `network` is matched and labeled twice, and the latter label covers the former label, resulting in a illegal label pair $\{B_{CW}, B_{CW}\}$ in BIOES scheme. 
Thus, to solve this problem, the first three words are labeled as $\{B_{CW}, E_{CW}, O\}$.
Perhaps you have a better way to address this issue, please raise it in the issues.

The annotations we used in our labeled data differ from those in the paper, and below are their corresponding relationships.

These are word-level labels:
> B-CW = $B_{kp}$
>
> I-CW = $I_{kp}$
> 
> E-CW = $E_{kp}$
> 
> S-SW = $SW$
> 
> O = $O$

These are phrase-level labels:
> CW = $MW$
>
> SW = $SW$
>
> O = $O$

If you downloaded our preprocessed data, you can skip the preprocess step.

You can also process other dataset with this command, but sure that the format is same as the file under `/datasets/copyrnn_datasets/kp20k_sorted`.

	python ./code/labeling.py -src_file_path [path of source file] -trg_file_path [path of target file] -sl_src_file_path [path of source file for sequence label]

# Training
To train the model.

	python ./code/train.py

# Test and Evaluation
We merge testing, post-process and evaluation in the eval.sh.

If you want to onlt test the model, you can execute the following command:

	python ./code/test.py -dataset_directorys ["kp20k", "nus", "inspec", "semeval"] -model_names ["paecrf"]

You can comment out lines 34 to 37 in `eval.sh`, and then execute the following command to perform post-processing and evaluate the results.

	./code/eval.sh "kp20k" "inspec" "nus" "semeval" "--" "paecrf" 









