## test librispeech dataset using torchaudio

import os
import sys

sys.path.append("/mnt/lustre/sjtu/home/zym22/huggingface/Llama-X")

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.distributed
import transformers
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from transformers import Trainer
from datasets import load_dataset
import torchaudio
import datasets

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

def train_tokenize_function(examples, tokenizer):
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    if 'input' in examples:
        sources = [
            prompt_input.format_map(dict(instruction=instruction, input=input)) if input != "" \
            else prompt_no_input.format_map(dict(instruction=instruction)) \
            for instruction, input in zip(examples['instruction'], examples['input']) 
        ]
    else:
        sources = [
            prompt_no_input.format_map(dict(instruction=instruction)) \
            for instruction in examples['instruction']
        ]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples['output']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/data/LM/llama-7b-hf",
        model_max_length=512,
        padding_side="right",
        use_fast=True,
    )

class FbankDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    def compute_fbank(self, audio, sample_rate, num_mel_bins=80, window_size=25, stride=10):
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform=audio,
            sample_frequency=sample_rate,
            num_mel_bins=num_mel_bins,
            frame_length=window_size,
            frame_shift=stride,
        )
        return fbank

    def __getitem__(self, index):
        data = self.dataset[index]
        audio_data, sample_rate = data[0], data[1]
        fbank_features = self.compute_fbank(audio_data, sample_rate)
        return {
            'audio': fbank_features,
            'sample_rate': data[1],
            'transcript': data[2],
            'speaker_id': data[3],
            'chapter_id': data[4],
            'utterance_id': data[5],
        }

def fbank_librispeech_collate_fn(batch, tokenizer):
    batch = [data for data in batch if data['audio'].shape[0] > 0]
    batch = sorted(batch, key=lambda x: x['audio'].shape[0], reverse=True)
    batch_size = len(batch)
    max_length = batch[0]['audio'].shape[0]
    demension = batch[0]['audio'].shape[1]

    audio = torch.zeros(batch_size, max_length, demension)
    audio_lens = torch.zeros(batch_size, dtype=torch.int32)
    transcript = []
    speaker_id = []
    chapter_id = []
    utterance_id = []
    for i, data in enumerate(batch):
        audio[i, :data['audio'].shape[0], :] = data['audio']
        audio_lens[i] = data['audio'].shape[0]
        transcript.append(data['transcript'])
        speaker_id.append(data['speaker_id'])
        chapter_id.append(data['chapter_id'])
        utterance_id.append(data['utterance_id'])
    tokenized = tokenizer(transcript, 
                          return_tensors="pt", 
                          padding="longest", 
                          max_length=tokenizer.model_max_length, 
                          truncation=True
                        )
    return {
        'audio': audio,
        'sample_rate': data['sample_rate'],
        'audio_lens': audio_lens,
        'tokenized': tokenized,
        'speaker_id': speaker_id,
        'chapter_id': chapter_id,
        'utterance_id': utterance_id,
    }


train_subsets = ["train-clean-100", "train-clean-360", "train-other-500"]
dev_subsets = ["dev-clean", "dev-other"]
test_subsets = ["test-clean", "test-other"]

raw_train_datasets = ConcatDataset(
    [
        torchaudio.datasets.LIBRISPEECH(
        root="/mnt/lustre/sjtu/shared/data/asr/rawdata", 
        folder_in_archive="LibriSpeech",
        url=subset, 
        download=False, 
        ) for subset in train_subsets
    ]
)

fbank_train_datasets = FbankDataset(raw_train_datasets)
fbank_train_dataloader = DataLoader(fbank_train_datasets, batch_size=32, shuffle=True, collate_fn=lambda x: fbank_librispeech_collate_fn(x, tokenizer), num_workers=4)