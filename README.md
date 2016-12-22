# Topic Modeling

The project contains implementations of topic models I have used during my thesis. 
Those implementations have been used for different papers. 

SentenceLDA
====

In particular, we have proposed the sentenceLDA in the SIGIR 2016 paper :
 [On a topic model for sentences](https://arxiv.org/pdf/1606.00253v1.pdf).
 
CopulaLDA
====
In particular, we have proposed copulaLDA in the Coling 2016 paper :
[Modeling topic dependencies in semantically coherent text spans with
copulas ](TBD).

References 
======

In case you use the model, please cite our paper:
```
@InProceedings{balikas2016sigir,
  author    = {Georgios Balikas and Massih-Reza Amini and Marianne Clausel},
  title     = {On a Topic Model for Sentences},
  booktitle = {Proceedings of the 39th International {ACM} {SIGIR} conference on Research and Development in Information Retrieval, {SIGIR} 2016, Pisa, Italy, July 17-21, 2016},
  pages     = {921--924},
  year      = {2016}}
```

For the copulaLDA model, please also cite:
```
@InProceedings{balikas-EtAl:2016:COLING,
  author    = {Balikas, Georgios  and  Amoualian, Hesam  and  Clausel, Marianne  and  Gaussier, Eric  and  Amini, Massih R},
  title     = {Modeling topic dependencies in semantically coherent text spans with copulas},
  booktitle = {Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers},
  month     = {December},
  year      = {2016},
  address   = {Osaka, Japan},
  publisher = {The COLING 2016 Organizing Committee},
  pages     = {1767--1776},
  url       = {http://aclweb.org/anthology/C16-1166}
}

```
Notes
====
This is development code and may not be fully functional. That said, the code was tested with Python 2.7 and R 3.1.1 and was functional. Normally, you should be able to reproduce all the experiments reported in the papers. 

