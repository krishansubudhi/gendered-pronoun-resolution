Official documentation
### Set up local env
https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment#local

Local environment will allow to submit code from local machine. Changes can be made in the IDE and will reflect.

I am configuring local env as I will need more iterations.

Or 

### Create a notebook VM
https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-1st-experiment-sdk-setup

This already sets up the environment for you. If you are starting out in a machine where nothing is installed already, this is a good place to start.

# Follow this guideline for end to end training example
https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-pytorch

# Hyperdrive

https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-tune-hyperparameters

## Bug in hyperdrive code


Fix this in file

C:\users\krkusuk\AppData\Local\Continuum\miniconda3\envs\pytorch\lib\site-packages\azureml\train\hyperdrive\runconfig.py
    
if isinstance(param,str) and param.lstrip("-") in parameter_space:

## Errors

1. Bert large with 64 batch size failing.
    Solution: dynamic batch size

2. concat models failing during argmax

        File "TrainGAP.py", line 132, in evaluate
            preds = torch.argmax(logits, dim = 1)
        RuntimeError: "argmax_cuda" not implemented for 'Half'
        Error occurred: User program failed with RuntimeError: "argmax_cuda" not implemented for 'Half'
    Solution : convert logits to float during argmax

### Observations

Bert large is giving 90% accuracy

# AML official documents


https://docs.microsoft.com/en-us/azure/architecture/reference-architectures/ai/training-deep-learning

https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-track-experiments

### Accessing data through datastores in AML compute

https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-access-data