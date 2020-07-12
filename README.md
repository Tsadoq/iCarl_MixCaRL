# iCaRL PyTorch Implementation and MixCaRL

Project for the course of Machine Learning and Deep Learning @ PoliTO.

## Introduction

Given the widespread presence of continuous streams of data in modern industry and the capillary deploy- ment of learning systems with limited power, the ability to learn in a continuous way would be greatly beneficial for present Machine Learning methods.
In this paper we followed multiple steps:
- Atfirst,we proceeded by computing a baseline,that is, we checked the effects of catastrophic forgetting: such thing was achieved by training the network on a batch of 10 classes.
- Similarly, we computed a benchmark by using the Joint Training strategy, that is, by retraining the net at each step with all the data available up to this point.
- We then implemented the Learning Without Forgetting method, the first academic step for Incremental learning. With this basic yet quite interesting strategy we saw the first improvement with respect to catastrophic forgetting.
- We then proceeded with the implementation of iCaRL, a famous strategy for Incremental Classification and Representation Learning.
- Finally, we proposed some variations to the standard iCaRL implementation that could be beneficial in some particular cases.

## MixCarl

Starting from the idea of better taking advantage of the exemplars, we tried experimenting with some ideas on how to preserve more features using the same number of exem- plars or how to achieve similar results while using less. The general idea behind the following studies is to combine im- ages at the lowest level possible, which is pixel by pixel, and then normalize according to the strategy we used.

When using few exemplars such method seems to outperform iCaRL implementation

## Docs
