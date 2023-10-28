# Variational Inference with EM algorithm

Library implementation of variational inference with em maximization on the task of polygenic risk prediction (PRS) from GWAS marginal statistics. This repository was created to accompany my two part blog post on variational inference.

Part 1: [Theory and motivation of variational inference with expectation maximization](https://matrixmaster.me/blog/2023/variational-inf-1/)
Part 2: [Implementation of variational inference on PRS posterior effect size inference](https://matrixmaster.me/blog/2023/variational-inf-2/)

To use the library, please update `main.py` by adding your desired functional prior distribution, likelihood distribution, and parameter update functions in the `compute_elbo()` function.
