import pandas as pd
import sklearn
import torch
from torch.utils.data import DataLoader
import jsonlines
import typing
from typing import Any, Dict, List, Optional
import datasets
from datasets import Dataset, load_dataset, DatasetDict
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import BackboneFinetuning, BaseFinetuning
from torch.optim.optimizer import Optimizer
from pytorch_lightning.callbacks import ModelPruning, EarlyStopping, ModelCheckpoint
import lightning_transformers
from lightning_transformers.core.nlp import HFBackboneConfig, HFTransformerDataConfig, HFDataModule
from lightning_transformers.task.nlp.text_classification import (
    TextClassificationDataModule, TextClassificationTransformer)
import transformers
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModelForSequenceClassification
from sklearn.metrics import classification_report



def create_sample(df: pd.DataFrame, indices: list, n: int = 2) -> list:
    """
        Creates samples from full dataframe using indices and n rows before.
    Returns a list of [joined texts from n rows before, current sentence].

    :param df: full dataframe
    :param indices: list of index numbers of dataframe to extract
    :param n: number of prior rows to return
    :return: a list of [joined texts from n rows before, current sentence]

    """
    samples = []
    for idx in indices:
        if idx <= n:
            continue

        samples.append([
            ' '.join(df.loc[idx - n:idx - 1, 'article'].to_list()),
            df.loc[idx, 'article']
        ])
    return samples

def create_df(data):
    # code to split it into 2 lists
    col1, col2 = map(list, zip(*data))

    df = pd.DataFrame(list(zip(col1, col2)),
                  columns=['descA','descB'])

    return df

# edited data module to only take in test data for prediction, as train set

class EntityMatchingDataModule(TextClassificationDataModule):
    def __init__(self,
                    cfg: HFTransformerDataConfig,
                    tokenizer: PreTrainedTokenizerBase,
                    train_data: pd.DataFrame):
        super().__init__(tokenizer, cfg)
        self.train_data = train_data

    def load_dataset(self) -> DatasetDict:
        return DatasetDict({
            'train': Dataset.from_pandas(self.train_data)})

    def process_data(self, dataset, stage: Optional[str] = None) -> Dataset:
        dataset = EntityMatchingDataModule.preprocess(
            dataset,
            tokenizer=self.tokenizer,
            padding=self.cfg.padding,
            truncation=self.cfg.truncation,
            max_length=self.cfg.max_length,
        )
        cols_to_keep = [
            x for x in ["input_ids", "attention_mask", "token_type_ids",'labels'] if x in dataset["train"].features
        ]
        dataset.set_format("torch", columns=cols_to_keep)
        self.labels = dataset["train"].features["labels"]
        self.labels.num_classes = 2
        return dataset

    @staticmethod
    def convert_to_features(
        example_batch: Any, _, tokenizer: PreTrainedTokenizerBase, **tokenizer_kwargs
    ):
    # for our use case, we will have to tokenize our 2 examples for entity matching
        return tokenizer(example_batch['descA'],
                            example_batch['descB'],
                            padding=True,
                            truncation=True)

    @staticmethod
    def preprocess(ds: Dataset, **fn_kwargs) -> Dataset:
        ds = ds.map(
            # todo: change this to self.convert_to_features for users to override
            EntityMatchingDataModule.convert_to_features,
            batched=True,
            with_indices=True,
            fn_kwargs=fn_kwargs,
        )
        ds.rename_column_("label", "labels")
        return ds

class EntityMatcher(TextClassificationTransformer):
    def __init__(self, learning_rate=1e-5, max_lr=1e-3,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # this is to initialize the backbone in this instance
        for k,v in kwargs.items():
            if k == 'backbone': 
                self.backbone = v

        self.lr = learning_rate
        self.max_lr = max_lr
        # self.num_classes = num_classes
        # self.save_hyperparameters()
    
    # this helps to hook num_classes to underlying model
    # without it, cannot call predict
    def setup(self, stage):
      self.model.num_classes=2
    
    def forward(self, x): # for inference
        # import pdb; pdb.set_trace()
        input_ids = x['input_ids']
        token_type_ids = x['token_type_ids']
        attention_mask = x['attention_mask']
        return self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)

