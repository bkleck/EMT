# EMT
Entity-Matching Transformer with BERT  
<img src='https://user-images.githubusercontent.com/77097236/119448365-d18ca700-bd63-11eb-8522-efa3766e2fff.png' width="500" height="250">


## Table of Contents

## Introduction
With the vast sources of articles for start-ups available online, much of the sentences do not refer to the entity/company we are interested in. Thus, this model aims to match sentences to the main entity we are interested in using a custom BERT architecture.   
Afterwards, the results will be passed to along the pipeline to other models to predict the presence and direction of various signals such as revenue, growth, etc...


## Technology
This was done on Google Colab, with the main libraries being pytorch-lightning and lightning-transformer.  
The trainer notebook is implemented on vanilla Pytorch, while the lightning_trainer notebook is implemented with lightning modules.  
These modules define pre-made tasks that can easily be configured and contain various functions to help accelerate and improve training.  

## Model Architecture
With the need to implement 2 separate sentences into the BERT embeddings, we created custom tokens, with reference to this paper:
https://openproceedings.org/2020/conf/edbt/paper_205.pdf
- Conventional tokens for BERT: [CLS] + text + [SEP]
- Custom entity tokens: [CLS] + A + [SEP] + B + [SEP]  
![image](https://user-images.githubusercontent.com/77097236/119447935-3562a000-bd63-11eb-987b-c9ea735e96f0.png)

