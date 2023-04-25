# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import logging
import math
import os

import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 7"
from dataclasses import dataclass, field
from typing import Optional
import argparse

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2Config,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def get_ml_dataset(tokenizer, file_path, block_size):
    return LineByLineTextDataset(tokenizer, file_path, block_size)


def fine_tune(parser):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    print("**************** begin fine-tune *****************")
    # Set seed
    set_seed(42)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # model_path = "./pretrained/gpt2"
    # train_path = "./dataset/clinc/clinc_train_gpt.txt"
    # eval_path = "./dataset/clinc/clinc_valid_gpt.txt"
    model_path = parser.pretrained_path
    train_path = parser.train_path
    eval_path = parser.eval_path
    num_epoch = parser.epoch


    config = GPT2Config.from_pretrained(model_path)

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    model = GPT2LMHeadModel.from_pretrained(model_path, config=config)

    block_size = 300
    special_tokens_dict = {'pad_token': '<PAD>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # Get datasets

    train_dataset = get_ml_dataset(tokenizer, train_path, block_size)
    eval_dataset = get_ml_dataset(tokenizer, eval_path, block_size)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, mlm_probability=0.15
    )
    '''
    python
    run_language_modeling.py 
    --train_data_file data / origin / glue_mnli_train.txt 
    --output_dir ckpts / gpt2 - glue_mnli 
    --model_type gpt2 
    --model_name_or_path gpt2 
    --save_total_limit 1 
    --num_train_epochs 1 
    --do_train 
    --per_gpu_train_batch_size 8 
    --per_gpu_eval_batch_size 8 
    --line_by_line 
    --gradient_accumulation_steps 1 
    --cache_dir. / datafortrain 
    --overwrite_output_dir
    '''
    output_path = parser.model_path
    print("  model path: {}".format(output_path))
    train_args = TrainingArguments(output_dir=output_path,
                                   evaluation_strategy="steps",
                                   eval_steps=500,
                                   num_train_epochs=num_epoch,
                                   overwrite_output_dir=True,
                                   save_total_limit=1,
                                   load_best_model_at_end=True,
                                   logging_steps = 500,
                                   logging_first_step=True)
    train_args.per_gpu_train_batch_size = 8
    train_args.do_eval = True
    print("   gpu num:", train_args.n_gpu)
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)

    tokenizer.save_pretrained(train_args.output_dir)


if __name__ == "__main__":
    pass
    #fine_tune()
