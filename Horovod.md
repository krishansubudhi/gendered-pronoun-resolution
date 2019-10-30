# Horovod

![Img](https://user-images.githubusercontent.com/16640218/38965607-bf5c46ca-4332-11e8-895a-b9c137e86013.png)

[Horovod Tutorial](https://github.com/horovod/horovod)

[MPI Tutorial](https://mpitutorial.com/tutorials/mpi-introduction/)

[Analyze all reduce performance](https://github.com/horovod/horovod/blob/master/docs/timeline.rst)

[Install openMPI, NCCL and Horovod](https://github.com/horovod/horovod/blob/master/docs/gpus.rst)

[Pytorch Tutorial](https://github.com/horovod/horovod/blob/master/docs/pytorch.rst)


One of the unique things about Horovod is its ability to interleave communication and computation coupled with the ability to batch small allreduce operations, which results in improved performance. We call this batching feature [Tensor Fusion](https://github.com/horovod/horovod/blob/master/docs/tensor-fusion.rst).

[Auto tune tensor fusion performance] (https://github.com/horovod/horovod/blob/master/docs/autotune.rst)

Horovod achieves 90% scaling efficiency for both Inception V3 and ResNet-101, and 68% scaling efficiency for VGG-16. 

## Multi node

Horovod requires every node to be able to ssh with each other without password. SSH uses port 22 for connection.
For passwordless connection

Node 1

     ssh-keygen -t rsa
     cat .ssh/id_rsa.pub


Node 2

     vim .ssh/authorized_keys
     write node 1 public key
     
Add the host itself in the authorized user list

    cat .ssh/id_rsa.pub >> .ssh/authorized_keys

Do same for node 2 - node 1 too

Test in node 1

    ssh <node2 ip>
prompt 'yes' for the first time. No prompt next time (verify later)


Test in node 2

    ssh <node2 ip>

https://github.com/horovod/horovod/blob/master/docs/running.rst

## How Horovod works?

Loss.backward hook is present in [DistributedOptimize](https://github.com/horovod/horovod/blob/master/horovod/torch/__init__.py) class


    def _make_hook(self, p)

The following registers all the hooks in the model parameters

    def _register_hooks(self)


Then during the stepping,horovod synchronizes all the grads

``` python
def step(self, closure=None):
    if self._should_synchronize:
        ---
        self.synchronize()
    self._synchronized = False
    return super(self.__class__, self).step(closure)
```

The class to synchronize does the all reduce operation

```python
def synchronize(self):
    for p, value in self._handles.items():
        handle, ctx = self._allreduce_grad_async(p)
```