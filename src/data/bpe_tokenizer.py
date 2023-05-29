from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class BPETokenizer:
    def __init__(self, sentence_list, max_len=15):
        """
        sentence_list - список предложений для обучения
        """
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.train_from_iterator(sentence_list, trainer=trainer)
        self.max_len = max_len

        self.word2index = dict()
        self.index2word = dict()
        self.mapper()

    def mapper(self):
        self.word2index = self.tokenizer.get_vocab()
        self.index2word = {v: k for k, v in self.word2index.items()}

    def __call__(self, sentence):
        """
        sentence - входное предложение
        """
        ids = self.tokenizer.encode(sentence).ids

        if len(ids) >= self.max_len:
            ids = ids[:self.max_len]
        else: 
            pad_idx = self.word2index["[PAD]"]
            len_diff = self.max_len - len(ids)
            ids = ids + [pad_idx] * len_diff

        return ids

    def decode(self, token_list):
        """
        token_list - предсказанные ID вашего токенизатора
        """
        return self.tokenizer.decode(token_list)