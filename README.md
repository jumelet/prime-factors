# Do Language Models Exhibit Human-like Structural Priming Effects?
Code for the ACL Findings paper "Do Language Models Exhibit Human-like Structural Priming Effects?"

## Code
The code for the extraction of Priming Effects and the paper's analysis are split into two folders: `extraction` and `analysis`.

PE extraction is done using the `transformers` and `diagnnose` libraries, and results in a `.csv` file containing the PE scores for each prime/target pair. This file can then be processed using `main.ipynb` notebook in the `extraction` folder, which contains the code for all the statistical experiments.

## PE Extraction
The PE scores for a particular LM can be extracted using the following command:

`python3 main.py --model $HF_MODEL_NAME --data $DATA_DIR --save $SAVE_DIR`

Where `$HF_MODEL_NAME` is a huggingface model name, `$DATA_DIR` points to the data directory found in `extraction/data` (the command creates scores for all files in that directory), and `$SAVE_DIR` a directory to which the scores file will be written. 

## Model Scores
The dataframe containing all our model scores can be downloaded here: https://drive.google.com/file/d/1vvvq8ASQgPRkpYb3Cykn2f7n1PWBbLOz/view?usp=sharing

Please reach out to me (Jaap) if you have any questions about the code or methodology!

## Citation
If you wish to cite our paper, you can use the following bib:
```@inproceedings{jumelet-etal-2024-language,
    title = "Do Language Models Exhibit Human-like Structural Priming Effects?",
    author = "Jumelet, Jaap  and
      Zuidema, Willem  and
      Sinclair, Arabella",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.877",
    pages = "14727--14742",
}
```
