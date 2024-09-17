# Do Language Models Exhibit Human-like Structural Priming Effects?
Code for the ACL Findings paper "Do Language Models Exhibit Human-like Structural Priming Effects?"

The code for the plots and experiments can be found in `src/main.ipynb`. The code for computing the priming effects can be found in `src/pe_scores.ipynb` (will be added soon).

The dataframe containing all our model scores can be downloaded here: https://drive.google.com/file/d/1vvvq8ASQgPRkpYb3Cykn2f7n1PWBbLOz/view?usp=sharing

Please reach out to me (Jaap) if you have any questions about the code or methodology!

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
    abstract = "We explore which linguistic factors{---}at the sentence and token level{---}play an important role in influencing language model predictions, and investigate whether these are reflective of results found in humans and human corpora (Gries and Kootstra, 2017). We make use of the structural priming paradigm{---}where recent exposure to a structure facilitates processing of the same structure{---}to investigate where priming effects manifest, and what factors predict them. We find these effects can be explained via the inverse frequency effect found in human priming, where rarer elements within a prime increase priming effects, as well as lexical dependence between prime and target. Our results provide an important piece in the puzzle of understanding how properties within their context affect structural prediction in language models.",
}```
