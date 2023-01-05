# KRL

Reproduceing the models of Knowledge Representation Learning (KRL), such as TransE, TransH etc.

Don't use this repo because I am developing it. So it must suffer broken changes in future.

## How to use it?

You can use it by cloning this repo and modify some codes to reproduce what you want.

The call to the training code has been warpped through [Typer](https://typer.tiangolo.com/), which is a great tools for building CLIs. So we can call the training code by CLIs. The examples can be found in directory `./examples`.

Example:

```shell
cd ./examples
sh transe.sh
```

+ Notice: Before you run this script, you should download the dataset, such as FB15k, and modify the script for choosing the path of dataset and checkpoints.

The example `./transe.ipynb` is a good tutorial for reproduce the TransE if you want to know the structure of this repo. This tutorial can be run without any dependencies, except for the use of common third-party libraries like PyTorch and Numpy.

## Plan

| Status |  Model   | Year | Paper  | Rewarks |
|  :----:  | :----:  | :----: | :--- | --- |
| :heavy_check_mark:  | RESCAL | 2011 | ICML'11, [OpenReview](https://openreview.net/forum?id=H14QEiZ_WS) | |
| :heavy_check_mark:  | TransE | 2013 | NIPS'13, [ACM](http://dl.acm.org/doi/10.5555/2999792.2999923) | |
| :heavy_check_mark:  | TransH | 2014 | AAAI'14, [ReasearchGate](https://www.researchgate.net/publication/319207032_Knowledge_Graph_Embedding_by_Translating_on_Hyperplanes) | |
| :heavy_check_mark: | DistMult | 2014 | ICLR'15, [arXiv](http://arxiv.org/abs/1412.6575) | |
| :heavy_check_mark: | TransR | 2015 | AAAI'15, [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/9491) | a low performance, but I don't know why. |
| :white_circle: | TransD | 2015 | ACL-IJCNLP 2015, [Aclanthology](https://aclanthology.org/P15-1067) | |
| :white_circle: | TransF | 2016 | AAAI'16, [AAAI](https://www.aaai.org/ocs/index.php/KR/KR16/paper/view/12887) | |
| :white_circle: | ComplEx | 2016 | ICML'16, [arXiv](http://arxiv.org/abs/1606.06357) | |
| :white_circle: | HolE | 2016 | AAAI'16, [arXiv](http://arxiv.org/abs/1510.04935) | |
| :white_circle: | R-GCN | 2017 | ESWC'18, [arXiv](http://arxiv.org/abs/1703.06103) | |
| :white_circle: | ConvKB | 2018 | NAACL-HLT 2018, [arXiv](http://arxiv.org/abs/1712.02121) | |
| :white_circle: | ConvE | 2018 | AAAI'18, [arXiv](http://arxiv.org/abs/1707.01476) | |
| :white_circle: | SimplE | 2018 | NIPS'18, [arXiv](http://arxiv.org/abs/1802.04868) | |
| :white_circle: | RotatE | 2019 | ICLR'19, [arXiv](http://arxiv.org/abs/1902.10197) | |
| :white_circle: | QuatE | 2019 | NeurIPS'19, [arXiv](http://arxiv.org/abs/1904.10281) | |
| :white_circle: | ConvR | 2019 | NAACL-HLT 2019, [Aclanthology](https://aclanthology.org/N19-1103) | |
| :white_circle: | KG-BERT | 2019 | [arXiv](http://arxiv.org/abs/1909.03193) | |
| :white_circle: | PairRE | 2021 | ACL-IJCNLP 2021, [Aclanthology](https://aclanthology.org/2021.acl-long.336) | |
