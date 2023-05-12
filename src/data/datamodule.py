from torch.utils.data import DataLoader

from data.mt_dataset import MTDataset
from data.space_tokenizer import SpaceTokenizer
from data.utils import TextUtils, short_text_filter_function
from data.bpe_tokenizer import BPETokenizer
from transformers import T5Tokenizer
import yaml

class DataManager:
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.tokenizer_type = config["tokenizer_type"]

    def prepare_data(self):
        pairs = TextUtils.read_langs_pairs_from_file(filename=self.config["filename"])
        prefix_filter = self.config['prefix_filter']
        if prefix_filter:
            prefix_filter = tuple(prefix_filter)

        source_sentences,target_sentences = [], []
        # dataset is ambiguous -> i lied -> я солгал/я соврала
        unique_sources = set()
        for pair in pairs:
            source, target = pair[0], pair[1]
            if short_text_filter_function(pair, self.config['max_length'], prefix_filter) and source not in unique_sources:
                source_sentences.append(source)
                target_sentences.append(target)
                unique_sources.add(source)

        train_size = int(len(source_sentences)*self.config["train_size"])
        source_train_sentences, source_val_sentences = source_sentences[:train_size], source_sentences[train_size:]
        target_train_sentences, target_val_sentences = target_sentences[:train_size], target_sentences[train_size:]

        if self.tokenizer_type == 'BPETokenizer':
            self.source_tokenizer = BPETokenizer(source_train_sentences)
            tokenized_source_train_sentences = [self.source_tokenizer(s) for s in source_train_sentences]
            tokenized_source_val_sentences = [self.source_tokenizer(s) for s in source_val_sentences]

            self.target_tokenizer = BPETokenizer(target_train_sentences)
            tokenized_target_train_sentences = [self.target_tokenizer(s) for s in target_train_sentences]
            tokenized_target_val_sentences = [self.target_tokenizer(s) for s in target_val_sentences]
            
        elif self.tokenizer_type == 'T5Tokenizer':
            self.source_tokenizer = T5Tokenizer.from_pretrained(self.config["pretrained_type"])
            self.target_tokenizer = self.source_tokenizer

            tokenized_source_train_sentences = self.source_tokenizer([s for s in source_train_sentences],
            padding="longest", max_length=self.config['max_length'], truncation=True).input_ids

            tokenized_source_val_sentences = self.source_tokenizer([s for s in source_val_sentences],
            padding="longest", max_length=self.config['max_length'], truncation=True).input_ids

            tokenized_target_train_sentences = self.source_tokenizer([s for s in target_train_sentences],
            padding="longest", max_length=self.config['max_length'], truncation=True).input_ids

            tokenized_target_val_sentences = self.source_tokenizer([s for s in target_val_sentences],
            padding="longest", max_length=self.config['max_length'], truncation=True).input_ids
            
            self.config['tokenizer_length'] = len(self.source_tokenizer) 
            self.config['tokenizer'] = self.source_tokenizer
            yaml.dump(self.config, stream=open("configs/t5_data_config.yaml", "w"))
        
        else:
            raise AttributeError('tokenizer_type is not properly defind')

        train_dataset = MTDataset(tokenized_source_list=tokenized_source_train_sentences,
                                  tokenized_target_list=tokenized_target_train_sentences, dev=self.device)

        val_dataset = MTDataset(tokenized_source_list=tokenized_source_val_sentences,
                                tokenized_target_list=tokenized_target_val_sentences, dev=self.device)

        train_dataloader = DataLoader(train_dataset, shuffle=True,
                                      batch_size=self.config["batch_size"],
        )

        val_dataloader = DataLoader(val_dataset, shuffle=True,
                                    batch_size=self.config["batch_size"], drop_last=True)
        return train_dataloader, val_dataloader
