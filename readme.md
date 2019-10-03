This is the solution to the famous Kaggle comptetion.
https://www.kaggle.com/c/gendered-pronoun-resolution/overview

The pytorch based solution utilizes BERT architecture to create a finetuned model.

Training and validation data: https://github.com/google-research-datasets/gap-coreference.git

Goals:

1. Create a working solution with good accuracy.
2. Compare different approaches.
3. Apply Distributed training using pytorch Distributed Data Parallel and Horovod. Document speed improvement.
4. Use NVIDIA apex library for 16 bit floating point precission (fp16). Show training speed and metrics.
5. Use azure ML to train. 
6. Use hyperdrive.
