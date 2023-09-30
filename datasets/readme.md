Please download datasets and place files and folders following the directory tree structure.

  ├─copyrnn_datasets
  │  ├─cross_domain_separated
  │  ├─cross_domain_sorted
  │  ├─kp20k_separated
  │  ├─kp20k_sorted
  │  └─sequence_label_datasets
  ├─inspec
  ├─kp20k
  ├─nus
  └─semeval

The data under the folder **sequence_label_datasets** is labeled for the sequence labeling task.
The labeling code is **/code/labeling.py**.
You can also process other dataset with this code, but sure that the format is same as the file under **kp20k_sorted**.

In the four dataset folders, we make a folder called **paecrf** to place the predictions made by PAE-CRF for this dataset.
