from utils import create_sample, create_df, EntityMatchingDataModule, EntityMatcher
import pandas as pd
import datetime
import pytorch_lightning as pl
import transformers
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModelForSequenceClassification
import lightning_transformers
from lightning_transformers.core.nlp import HFBackboneConfig, HFTransformerDataConfig, HFDataModule
import torch

# please make sure GPU is enabled with CUDA on pytorch
# https://varhowto.com/install-pytorch-cuda-10-1/
print('Is your GPU enabled?')
print(torch.cuda.is_available())

begin_time = datetime.datetime.now() # track run-time

# create dataframe to be loaded for prediction
df = pd.read_csv('data/news_tagged.csv', encoding='cp1252')
index_list = df.index.tolist()

data = create_sample(df, index_list, 2)
train_data = create_df(data)
train_data['label'] = 0

# Run predictions
model = EntityMatcher.load_from_checkpoint(checkpoint_path = 'bert_final.ckpt')

tokenizer = AutoTokenizer.from_pretrained(
    'bert-base-uncased')

data_loader = EntityMatchingDataModule(
    cfg=HFTransformerDataConfig(
        # num_workers=12,
        batch_size=8, # keep to max of 8, only use 16 with colab pro
        max_length=512),
    tokenizer=tokenizer,
    train_data=train_data)

trainer = pl.Trainer(
    gpus=1,
    precision=16, # change from fp32 to 16 for faster run-time
    progress_bar_refresh_rate=20, # slow down refresh rate for colab
)

# call setup to initiate data loader without training step -- for inference
data_loader.setup()
# data is loaded under train, although is test set
test_loader = data_loader.train_dataloader()

predicted_values = trainer.predict(model, test_loader)

# save labels from predictions
predictions = torch.cat([p.logits for p in predicted_values], dim=0)
preds = predictions.softmax(dim=1).argmax(dim=1).cpu()
train_data['preds'] = preds

train_data.to_csv('data/predictions.csv')
print('prediction.csv has been saved to data folder!')
print(' ')
print(datetime.datetime.now() - begin_time)