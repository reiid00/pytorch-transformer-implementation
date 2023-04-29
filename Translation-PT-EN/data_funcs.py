from enum import Enum
from torch.utils.data import Dataset
import itertools
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.utils.data import random_split

NUM_PHRASES = 1_000_000
OUTPUT_FILE = 'data/en-pt_sentences.txt'
TOKENIZER_DIR = 'tokenizer'
VOCAB_SIZE = 64_000
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = '<unk>'


class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return torch.tensor(src), torch.tensor(tgt)

class LanguageDirection(Enum):
    PT2EN = 0,
    EN2PT = 1

def load_dataset(dataset_path, language_direction, limit=1000000):
    with open(dataset_path, "r", encoding="utf-8") as file:
        if language_direction == LanguageDirection.PT2EN.name:
            sentence_pairs = [tuple(reversed(line.strip().split("\t"))) for line in itertools.islice(file, limit)]
        else:
            sentence_pairs = [tuple(line.strip().split("\t")) for line in itertools.islice(file, limit)]
    return sentence_pairs

def preprocess_text(text):
    # Convert the text to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-záàâãéèêíïóôõöúçñ]+', ' ', text)

    return text

def add_special_tokens(tokens, bos_token, eos_token):
    return [bos_token] + tokens + [eos_token]


def yield_tokens(pairs, language, src_language, tgt_language):
    for src_tokens, tgt_tokens in pairs:
        if language == src_language:
            yield src_tokens
        elif language == tgt_language:
            yield tgt_tokens

def padded_sequence(sequence, max_length, pad_idx):
    return sequence + [pad_idx] * (max_length - len(sequence))


def collate_fn(batch, pad_idx_src, pad_idx_tgt):
    src_tensors, tgt_tensors = zip(*batch)
    src_tensors = pad_sequence(src_tensors, batch_first=True, padding_value=pad_idx_src)
    tgt_tensors = pad_sequence(tgt_tensors, batch_first=True, padding_value=pad_idx_tgt)
    return src_tensors, tgt_tensors


def load_data(dataset_path, language_direction = LanguageDirection.PT2EN.name, limit = 1_000_000, batch_size = 32, max_len = 256):
    language_direction = LanguageDirection[language_direction]

    sentence_pairs = load_dataset(dataset_path, language_direction, limit=limit)

    en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
    pt_tokenizer = get_tokenizer("spacy", language="pt_core_news_sm")

    if language_direction == LanguageDirection.PT2EN:
        src_language, tgt_language = "pt", "en"
        src_tokenizer, tgt_tokenizer = pt_tokenizer, en_tokenizer
    else:
        src_language, tgt_language = "en", "pt"
        src_tokenizer, tgt_tokenizer = en_tokenizer, pt_tokenizer
    
    preprocessed_pairs = [(preprocess_text(src), preprocess_text(tgt)) for src, tgt in sentence_pairs]

    tokenized_pairs = [(
        src_tokenizer(src),
        tgt_tokenizer(tgt)
    ) for src, tgt in preprocessed_pairs]

    tokenized_pairs = [(
        add_special_tokens(src_tokens, BOS_TOKEN, EOS_TOKEN),
        add_special_tokens(tgt_tokens, BOS_TOKEN, EOS_TOKEN)
    ) for src_tokens, tgt_tokens in tokenized_pairs]

    tokenized_pairs = [(src_tokens, tgt_tokens) for src_tokens, tgt_tokens in tokenized_pairs if len(src_tokens) <= max_len and len(tgt_tokens) <= max_len]

    src_vocab = build_vocab_from_iterator(yield_tokens(tokenized_pairs, src_language, src_language, tgt_language), specials=[UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    tgt_vocab = build_vocab_from_iterator(yield_tokens(tokenized_pairs, tgt_language, src_language, tgt_language), specials=[UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])

    index_pairs = [(
        [src_vocab[token] for token in src_tokens],
        [tgt_vocab[token] for token in tgt_tokens]
    ) for src_tokens, tgt_tokens in tokenized_pairs]

    pad_idx_src = src_vocab[PAD_TOKEN]
    pad_idx_tgt = tgt_vocab[PAD_TOKEN]

    padded_pairs = [(
        padded_sequence(src_indices, max_len, pad_idx_src),
        padded_sequence(tgt_indices, max_len, pad_idx_tgt)
    ) for src_indices, tgt_indices in index_pairs]

    dataset = TranslationDataset(padded_pairs)

    # Calculate the sizes of the training and testing sets
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)  # Use 80% of the dataset for training
    test_size = dataset_size - train_size

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for the training and testing sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_fn, pad_idx_src=pad_idx_src, pad_idx_tgt=pad_idx_tgt))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_fn, pad_idx_src=pad_idx_src, pad_idx_tgt=pad_idx_tgt))

    return train_dataloader, test_dataloader, pad_idx_src, pad_idx_tgt, src_vocab, tgt_vocab


