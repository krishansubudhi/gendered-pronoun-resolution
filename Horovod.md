## Horovod

![Img](https://user-images.githubusercontent.com/16640218/38965607-bf5c46ca-4332-11e8-895a-b9c137e86013.png)

[Horovod Tutorial](https://github.com/horovod/horovod)

[MPI Tutorial](https://mpitutorial.com/tutorials/mpi-introduction/)

[Analyze all reduce performance](https://github.com/horovod/horovod/blob/master/docs/timeline.rst)

[Install openMPI, NCCL and Horovod](https://github.com/horovod/horovod/blob/master/docs/gpus.rst)

[Pytorch Tutorial](https://github.com/horovod/horovod/blob/master/docs/pytorch.rst)


One of the unique things about Horovod is its ability to interleave communication and computation coupled with the ability to batch small allreduce operations, which results in improved performance. We call this batching feature [Tensor Fusion](https://github.com/horovod/horovod/blob/master/docs/tensor-fusion.rst).

[Auto tune tensor fusion performance] (https://github.com/horovod/horovod/blob/master/docs/autotune.rst)

Horovod achieves 90% scaling efficiency for both Inception V3 and ResNet-101, and 68% scaling efficiency for VGG-16. 