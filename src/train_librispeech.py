#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List

import torch
import torch.distributed
import transformers
from torch.utils.data import Dataset, ConcatDataset
from transformers import Trainer
from datasets import load_dataset
import utils
import torchaudio

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


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


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


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

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

        def __len__(self):
            return len(self.dataset)

class DataCollatorForLibrispeechDataset():
    tokenizer: transformers.PreTrainedTokenizer
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        print(batch)
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
        tokenized = self.tokenizer(transcript, 
                            return_tensors="pt", 
                            padding="longest", 
                            max_length=self.tokenizer.model_max_length, 
                            truncation=True
                            )
        return dict(
            audio=audio,
            audio_lens=audio_lens,
            sample_rate=data['sample_rate'],
            tokenized=tokenized,
            speaker_id=speaker_id,
            chapter_id=chapter_id,
            utterance_id=utterance_id,
        )
        # return {
        #     'audio': audio,
        #     'sample_rate': data['sample_rate'],
        #     'audio_lens': audio_lens,
        #     'tokenized': tokenized,
        #     'speaker_id': speaker_id,
        #     'chapter_id': chapter_id,
        #     'utterance_id': utterance_id,
        # }
              
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    train_subsets = ["train-clean-100", "train-clean-360", "train-other-500"]
    dev_subsets = ["dev-clean", "dev-other"]
    test_subsets = ["test-clean", "test-other"]

    raw_train_datasets = ConcatDataset(
        [
            torchaudio.datasets.LIBRISPEECH(
            root=data_args.data_path, 
            folder_in_archive="LibriSpeech",
            url=subset, 
            download=False, 
            ) for subset in train_subsets
        ]
    )
    fbank_train_datasets = FbankDataset(raw_train_datasets)

    raw_dev_datasets = ConcatDataset(
        [
            torchaudio.datasets.LIBRISPEECH(
            root=data_args.data_path, 
            folder_in_archive="LibriSpeech",
            url=subset, 
            download=False, 
            ) for subset in dev_subsets
        ]
    )
    fbank_dev_datasets = FbankDataset(raw_dev_datasets)


    if training_args.local_rank > 0: 
        torch.distributed.barrier()

    if training_args.local_rank == 0:
        torch.distributed.barrier()
    
    if training_args.local_rank == 0:
        print("Training/evaluation parameters %s", training_args)
        print("Model parameters %s", model_args)
        print("Data parameters %s", data_args)
        print("Training dataset length: %d", len(fbank_train_datasets))
        print("Validation dataset length: %d", len(fbank_dev_datasets))
    
    data_collator = DataCollatorForLibrispeechDataset(tokenizer=tokenizer)

    #Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=fbank_train_datasets,
        eval_dataset=fbank_dev_datasets,
        tokenizer=tokenizer,
    )
    model.config.use_cache = False

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
