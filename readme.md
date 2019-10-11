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

Steps:

1. Boot a machine with at least one GPU

1. Clone the repo
  
  https://github.com/krishansubudhi/gendered-pronoun-resolution.git
  
2. cd into the cloned repo
3. Download data

  git clone https://github.com/google-research-datasets/gap-coreference.git
  
4. Preprocess: Convert text to tokens. Also find P , A and B position. Also add labels. The preprocessed data is stored as both tsv and pkl files. pkl file will be loaded for traning while tsv file is for readablility.

  python PreprocessGapData.py

5. Run code in single GPU with default configurations.

  python TrainGAP.py
