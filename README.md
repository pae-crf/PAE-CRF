The official implementation of paper "A Phrase-level Attention Enhanced CRF for Keyphrase Extraction".

# Code
This code is mainly adapted from [bert4torch](https://github.com/Tongjilibo/bert4torch). Thanks for his work.

# Quick Start
The whole process includes the following steps: Preprocess, Training, Evaluation.

The original datasets is provided by [kenchan](https://github.com/kenchan0226/keyphrase-generation-rl).

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




The directory structure of the entire project is as follows:



