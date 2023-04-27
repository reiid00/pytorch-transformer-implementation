import random
from torch.utils.data import Dataset, DataLoader
import itertools
import re
from tokenizers import ByteLevelBPETokenizer
import os
from torch.nn.utils.rnn import pad_sequence
import torch

FILE_PATH = 'data/en-pt.txt'
NUM_PHRASES = 1_000_000
OUTPUT_FILE = 'data/en-pt_sentences.txt'
TOKENIZER_DIR = 'tokenizer'
VOCAB_SIZE = 32_000

def load_dataset(dataset_path, limit=1000000):
    """
    Loads the dataset from the given path.

    Args:
        dataset_path: The path to the dataset file.
        limit: The maximum number of sentence pairs to load (default: 1000000).

    Returns:
        A list of (source, target) sentence pairs.
    """
    with open(dataset_path, "r", encoding="utf-8") as file:
        sentence_pairs = [tuple(line.strip().split("\t")) for line in itertools.islice(file, limit)]

    return sentence_pairs

def preprocess_text(text):
    """
    Preprocesses the given text.

    Args:
        text: The input text.
        remove_stopwords: Whether to remove stopwords from the text (default: True).

    Returns:
        A list of preprocessed words.
    """
    # Convert the text to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-záàâãéèêíïóôõöúçñ]+', ' ', text)

    return text

def train_tokenizer(sentence_pairs,temp_sentences_path, vocab_size=VOCAB_SIZE, min_frequency=2, output_dir=TOKENIZER_DIR):
    """
    Trains a tokenizer on the given sentence pairs.

    Args:
        sentence_pairs: The list of (source, target) sentence pairs.
        vocab_size: The vocabulary size for the tokenizer.
        min_frequency: The minimum frequency for a token to be included in the vocabulary.
        output_dir: The directory to save the tokenizer files.
    """
    # Save all sentences to a temporary file
    with open(temp_sentences_path, "w", encoding="utf-8") as file:
        for src, tgt in sentence_pairs:
            file.write(src + "\n")
            file.write(tgt + "\n")

    # Train the tokenizer
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[temp_sentences_path], vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    # Save the tokenizer
    tokenizer.save_model(output_dir)


class TranslationDataset(Dataset):
    def __init__(self, sentence_pairs, tokenizer, max_length):
        self.sentence_pairs = sentence_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        source, target = self.sentence_pairs[idx]
        tokenized_source = self.tokenizer.encode(source).ids[:self.max_length]
        tokenized_target = self.tokenizer.encode(target).ids[:self.max_length]
        return tokenized_source, tokenized_target

def load_tokenizer(tokenizer_dir=TOKENIZER_DIR):
    """
    Loads a tokenizer from the specified directory.

    Args:
        tokenizer_dir: The directory containing the tokenizer files.

    Returns:
        A Tokenizer object.
    """
    tokenizer = ByteLevelBPETokenizer(
        f"{tokenizer_dir}{os.sep}vocab.json",
        f"{tokenizer_dir}{os.sep}merges.txt",
        add_prefix_space=True
    )
    return tokenizer

def preprocess_data(sentence_pairs, tokenizer, max_length):
    """
    Preprocesses the sentence pairs by tokenizing and creating a TranslationDataset.

    Args:
        sentence_pairs: A list of (source, target) sentence pairs.
        tokenizer: The tokenizer used for tokenization.
        max_length: The maximum sequence length.

    Returns:
        A TranslationDataset object.
    """
    dataset = TranslationDataset(sentence_pairs, tokenizer, max_length)
    return dataset

def create_dataloader(dataset, batch_size,tokenizer, shuffle=True, num_workers=0):
    """
    Creates a DataLoader for the given dataset.

    Args:
        dataset: A PyTorch Dataset object.
        batch_size: The batch size to use in the DataLoader.
        shuffle: Whether to shuffle the dataset before creating the DataLoader.
        num_workers: The number of worker processes to use for loading the data.

    Returns:
        A DataLoader object.
    """
    def collate_fn(batch):
        src_tensors, tgt_tensors = zip(*batch)
        src_tensors = [torch.tensor(src) for src in src_tensors]
        tgt_tensors = [torch.tensor(tgt) for tgt in tgt_tensors]
        src_tensors = pad_sequence(src_tensors, batch_first=True, padding_value=tokenizer.token_to_id("<pad>"))
        tgt_tensors = pad_sequence(tgt_tensors, batch_first=True, padding_value=tokenizer.token_to_id("<pad>"))
        return src_tensors, tgt_tensors

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return dataloader