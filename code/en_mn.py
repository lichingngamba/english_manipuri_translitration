# import required library
import json
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import AutoTokenizer, AutoModel

# max_length
max_length = 512

# laod custom manipuri tokenizer
def mn_tokenizer(manipuri_text):
    # Tokenize the Manipuri text using the Byte-level BPE Tokenizer and the BertProcessing processor.
    tokenizer = ByteLevelBPETokenizer(
        "./meitei_tokenizer/meitei_tokenizer-vocab.json",
        "./meitei_tokenizer/meitei_tokenizer-merges.txt",
                                    )
    
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")), # sepearation token
        ("<s>", tokenizer.token_to_id("<s>")), # cls token
    )
    tokenizer.enable_truncation(max_length=max_length) # max length of the sequence.

    return (tokenizer.encode(manipuri_text, add_special_tokens=True))

# tokenizer decoder
def mn_decoder(manipuri_text_ids):
    # Tokenize the Manipuri text using the Byte-level BPE Tokenizer and the BertProcessing processor.
    tokenizer = ByteLevelBPETokenizer(
        "../meitei_tokenizer/meitei_tokenizer-vocab.json",
        "../meitei_tokenizer/meitei_tokenizer-merges.txt",
                                    )
    return tokenizer.decode(manipuri_text_ids)

# load english tokenizer and model
eng_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModel.from_pretrained("google/flan-t5-base")

# load train dataset
data = pd.read_csv("./Dataset/meitei_eng/train.csv")

# mi = preprocess_function(data, eng_tokenizer)
# print(mi)
# split data into train val
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=False)

# curate a token dataset.
def preprocess_function(dataset, en_tokenizer):
    inputs = [dataset["eng"].iloc[ex] for ex in dataset.index]
    targets = [dataset["mani"].iloc[ex] for ex in dataset.index]
 
    model_inputs = en_tokenizer(inputs, max_length=max_length, truncation=True)
    labels = list()
    for target in targets:
        # print(target)
        label = mn_tokenizer(target)
        ids = label.ids
        labels.append(ids)

    print("Length of input sequences:", len(model_inputs["input_ids"]))
    print("Length of labels:", len(labels))  
    model_inputs["labels"] = labels
    
    return model_inputs
#
val_data.reset_index(inplace= True) # reset index for customdata loader
train_set = preprocess_function(train_data, eng_tokenizer)
val_set = preprocess_function(val_data, eng_tokenizer)

# Model training
# loading dataset using custom data loader
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }
        
train_X = CustomDataset(input_ids= train_set['input_ids'], attention_mask= train_set['attention_mask'], labels= train_set['labels'])
val_X = CustomDataset(input_ids= val_set['input_ids'], attention_mask= val_set['attention_mask'], labels= val_set['labels'])

# Set up training arguments
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./meitei_eng/output", # output directory for saving model checkpoints and logs.
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    evaluation_strategy="epoch", # evaluate after every epoch.
    save_total_limit=2,
    eval_steps=500,
    logging_steps=500,
    learning_rate=5e-5,
    save_steps=500,
)

# Instantiate Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_X,
    eval_dataset=val_X
)

trainer.train()