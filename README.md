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

| Status |  Model   | Year | Paper  |
|  :----:  | :----:  | :----: | :--- |
| :heavy_check_mark:  | RESCAL | 2011 | ICML'11, [OpenReview](https://openreview.net/forum?id=H14QEiZ_WS) |
| :heavy_check_mark:  | TransE | 2013 | NIPS'13, [ACM](http://dl.acm.org/doi/10.5555/2999792.2999923) |
| :heavy_check_mark:  | TransH | 2014 | AAAI'14, [ReasearchGate](https://www.researchgate.net/publication/319207032_Knowledge_Graph_Embedding_by_Translating_on_Hyperplanes) |
| :heavy_check_mark: | DistMult | 2014 | ICLR'15, [arXiv](http://arxiv.org/abs/1412.6575) |
| :white_circle: | TransR | 2015 | AAAI'15, [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/9491) |
| :white_circle: | TransD | 2015 | ACL-IJCNLP 2015, [Aclanthology](https://aclanthology.org/P15-1067) |
| :white_circle: | ComplEx | 2016 | ICML'16, [arXiv](http://arxiv.org/abs/1606.06357) |
