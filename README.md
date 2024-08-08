# AGMR_entropy_sparse
# Semi-supervised Manifold Regularization with Adaptive Graph Construction

This repository contains the Matlab implementation of the paper titled "Semi-supervised manifold regularization with adaptive graph construction" published in "Pattern Recognition Letters". The paper can be accessed [here](http://dx.doi.org/10.1016/j.patrec.2017.09.004).

## Introduction
The goal of this project is to provide a semi-supervised learning algorithm that incorporates manifold regularization and adaptive graph construction. The algorithm is designed to handle both binary and multiclass classification problems.

## Code and Datasets
The code for this implementation can be found in the [AGMR_entropy_sparse](https://github.com/devn913/tree/main/AGMR_entropy_sparse) directory. The repository also includes datasets that can be used for experimentation.

## Flavors of Algorithm
The algorithm implemented in this project supports both binary and multiclass classification tasks. It can be used for multilabel learning in a semi-supervised setting.

## Usage
To use the multiclass algorithm for multilabel learning, follow these steps:
1. Clone the repository.
2. Install the required dependencies.
3. Load the desired dataset.
4. Preprocess the data if necessary.
5. Run the algorithm on the labeled and unlabeled data.
6. Evaluate the performance of the algorithm using appropriate metrics.

For more detailed instructions, please refer to the documentation provided in the repository.

## Citation


```
@ARTICLE{Wang2017-jd,
  title     = "Semi-supervised manifold regularization with adaptive graph
               construction",
  author    = "Wang, Yunyun and Meng, Yan and Li, Yun and Chen, Songcan and Fu,
               Zhenyong and Xue, Hui",
  abstract  = "Manifold regularization (MR) provides a powerful framework for
               semi-supervised classification (SSC) learning. It imposes the
               smoothness constraint over a constructed manifold graph, and its
               performance largely depends on such graph. However, 1) The
               manifold graph is usually pre-constructed before classification,
               and fixed during the classification learning process. As a
               result, independent with the subsequent classification, the
               graph does not necessarily benefit the classification
               performance. 2) There are parameters needing tuning in the graph
               construction, while parameter selection in semi-supervised
               learning is still an open problem currently, which sets up
               another barrier for constructing a ``well-performing'' manifold
               graph benefiting the performance. To address those issues, we
               develop a novel semi-supervised manifold regularization with
               adaptive graph (AGMR for short) in this paper by integrating the
               graph construction and classification learning into a unified
               framework. In this way, the manifold graph along with its
               parameters will be optimized in learning rather than
               pre-defined, consequently, it will be adaptive to the
               classification, and benefit the performance. Further, by
               adopting the entropy and sparse constraints respectively for the
               graph weights, we derive two specific methods called
               AGMR\_entropy and AGMR\_sparse, respectively. Our empirical
               results show the competitiveness of those AGMRs compared to MR
               and some of its variants.",
  journal   = "Pattern Recognit. Lett.",
  publisher = "Elsevier BV",
  volume    =  98,
  pages     = "90--95",
  month     =  oct,
  year      =  2017,
  language  = "en"
}
```

