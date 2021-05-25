# EMT
**Entity-Matching Transformer with BERT**  
<img src='https://user-images.githubusercontent.com/77097236/119448365-d18ca700-bd63-11eb-8522-efa3766e2fff.png' width="500" height="250">


## Table of Contents
[Introduction](#introduction) 
[Technology](#technology)


## Introduction
With the vast sources of articles for start-ups available online, much of the sentences do not refer to the entity/company we are interested in. Thus, this model aims to **_match sentences to the main entity_** we are interested in using a custom BERT architecture.   
Afterwards, the results will be passed to along the pipeline to other models to predict the presence and direction of various signals such as revenue, growth, etc...


## Technology
This was done on Google Colab, with the main libraries being `pytorch-lightning` and `lightning-transformer`.  
The `trainer` notebook is implemented on vanilla Pytorch, while the `lightning_trainer` notebook is implemented with lightning modules.  
These modules define pre-made tasks that can easily be configured and contain various functions to help accelerate and improve training.  

## Model Architecture
I made use of the **_AutoModel_** and **_AutoTokenizer_** from HuggingFace backbone to easily switch between different transformer architectures without the need to change the tokenization manually.

With the need to implement 2 separate sentences into the BERT embeddings, I created custom tokens, with reference to this paper:
https://openproceedings.org/2020/conf/edbt/paper_205.pdf
- Conventional tokens for BERT: **[CLS] + text + [SEP]**
- Custom entity tokens: **[CLS] + A + [SEP] + B + [SEP]**  
![image](https://user-images.githubusercontent.com/77097236/119447935-3562a000-bd63-11eb-987b-c9ea735e96f0.png)

Thus, for our data pre-processing, the data is split into samples in the following form: **['prior sentence' , 'current sentence']**.  
This was after experimentation between 1,2 and 3 prior sentences, where 1 prior sentence gave the best result.  
The labels for the data are **_is_entity_** and **_not_entity_**.

For detailed explanation on Lightning's `DataModule`, `TextClassificationTransformer` and `Trainer`, please refer to notebook, or Pytorch Lightning's [official documentation](https://pytorch-lightning.readthedocs.io/en/latest/).

## Model Finetuning
I experimented with various methods and models, results will be shared below.

### Finetuning Methods
**1) Stochastic Weight Averaging (SWA)**  
  This is an ensemble method for deep learning. It utilizes only 2 models: 1 with running average of weights, the other exploring with cyclic learning rate scheduler. It will then update the weights of the first model after each cycle.
  ![image](https://user-images.githubusercontent.com/77097236/119450611-c0916500-bd66-11eb-92c3-d56e79845da4.png)

**2) Model Pruning**  
  This is a compression technique to eliminate weights that contribute little to performance. It results in models being much smaller, but there might be a slight trade-off with accuracy.  
  ![image](https://user-images.githubusercontent.com/77097236/119450717-ea4a8c00-bd66-11eb-8334-95077a2c80ab.png)  

Other methods that we are exploring include **Auto LR Finder, Auto Batch-size and Base Finetuning**.  
These are the **_results_** from our experimentation:  
![image](https://user-images.githubusercontent.com/77097236/119451144-7eb4ee80-bd67-11eb-81d7-0bdeb492d3dd.png)    
These were run on Tesla P100 GPU. Do note that a batch-size of 16 can only be done on Colab Pro with high-RAM. As seen, both methods utilized turned out well, but running more epochs do not improve accuracy due to over-fitting. Thus, I will be adding an **_early-stopping callback_** later.  

### Model Experimentation
I also tried out various pre-trained transformers available on [HuggingFace](https://huggingface.co/transformers/pretrained_models.html).  
The **_results_** are shown below:  
![image](https://user-images.githubusercontent.com/77097236/119455808-8b881100-bd6c-11eb-981e-8aa95ded542c.png)

With more epochs together with early-stopping, I was able to push up the test F1 score. For the last 3 models, I utilized a Tesla V100 to speed up run-time significantly due to large model sizes.

## Results
After experimentation, we introduced a larger dataset into our final BERT model:
- **Callbacks: early-stopping, SWA, pruning**
- **9 epochs, 9min 55s**
- **Validation accuracy: 85.1%**

The `lightning-transformer` module was able to significantly reduce training speed, from **_34min 12s_** to **_9min 55s_**.

