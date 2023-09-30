Please download datasets and place files and folders following the directory tree structure.

	├─copyrnn_datasets
	│  ├─cross_domain_separated
	│  ├─cross_domain_sorted
	│  ├─kp20k_separated
	│  ├─kp20k_sorted
	│  └─sequence_label_datasets
	├─inspec
	│  └─paecrf
	│      ├─all
	│      ├─ck
	│      └─sk
	├─kp20k
	│  └─paecrf
	│      ├─all
	│      ├─ck
	│      └─sk
	├─nus
	│  └─paecrf
	│      ├─all
	│      ├─ck
	│      └─sk
	└─semeval
	    └─paecrf
	        ├─all
	        ├─ck
	        └─sk

The data under the folder `sequence_label_datasets` is labeled for the sequence labeling task.

In the four dataset folders, we make a folder called `paecrf` to place the predictions made by PAE-CRF for this dataset.

Our results is devided into three type of keyphrases, including all (i.e. allkeyword), ck(i.e. complexkeyword) and sk(i.e. simplekeyword).
