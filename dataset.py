import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict
import warnings

import spacy
from datasets import load_dataset

# ══════════════════════════════════════════════════════════════════════
# GLOBAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']

class Multi30kDataset(Dataset):
    def __init__(self, split: str = 'train', min_freq: int = 2):
        """
        Loads the Multi30k dataset and prepares tokenizers.

        Args:
            split    : 'train', 'validation', or 'test'.
            min_freq : Minimum frequency for a token to be included in the vocab.
        """
        self.split = split
        self.min_freq = min_freq

        # 1. Vocab structures (Populated by build_vocab)
        self.src_vocab: List[str] = []
        self.tgt_vocab: List[str] = []
        self.src_stoi: Dict[str, int] = {}
        self.tgt_stoi: Dict[str, int] = {}
        self.src_itos: Dict[int, str] = {}
        self.tgt_itos: Dict[int, str] = {}

        # 2. Final integer tensors (Populated by process_data)
        self.data: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # 3. Load raw dataset to prevent data leakage across splits
        print(f"[dataset] Loading Multi30k — split='{split}' ...")
        self.raw = load_dataset("bentrevett/multi30k", split=split)

        # 4. Load SpaCy models
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
            self.spacy_en = spacy.load("en_core_web_sm")
        except OSError as e:
            raise OSError(
                "spaCy language models not found. Run:\n"
                "  python -m spacy download de_core_news_sm\n"
                "  python -m spacy download en_core_web_sm"
            ) from e

        # 5. Tokenize all sentences ONCE and cache them in memory.
        print(f"[dataset] Tokenizing {len(self.raw)} pairs with SpaCy in memory...")
        self.src_tokens: List[List[str]] = [
            [tok.text.lower() for tok in self.spacy_de.tokenizer(ex['de'])]
            for ex in self.raw
        ]
        self.tgt_tokens: List[List[str]] = [
            [tok.text.lower() for tok in self.spacy_en.tokenizer(ex['en'])]
            for ex in self.raw
        ]
        print("[dataset] Tokenization complete.")

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        if self.split != 'train':
            warnings.warn("You are building vocabulary on a non-train split. This may cause data leakage.")
            
        print(f"[dataset] Building vocabularies (min_freq={self.min_freq}) ...")
        
        # Explicit frequency counting dictionaries
        src_freq: Dict[str, int] = {}
        tgt_freq: Dict[str, int] = {}

        # Count source tokens
        for sentence in self.src_tokens:
            for token in sentence:
                if token in src_freq:
                    src_freq[token] += 1
                else:
                    src_freq[token] = 1

        # Count target tokens
        for sentence in self.tgt_tokens:
            for token in sentence:
                if token in tgt_freq:
                    tgt_freq[token] += 1
                else:
                    tgt_freq[token] = 1

        # Initialize with special tokens
        self.src_vocab = list(SPECIAL_TOKENS)
        self.tgt_vocab = list(SPECIAL_TOKENS)

        # Append tokens meeting the frequency threshold
        for token, count in src_freq.items():
            if count >= self.min_freq:
                self.src_vocab.append(token)
                
        for token, count in tgt_freq.items():
            if count >= self.min_freq:
                self.tgt_vocab.append(token)

        # Sort the vocabularies (excluding special tokens) for deterministic ordering
        self.src_vocab = list(SPECIAL_TOKENS) + sorted(self.src_vocab[4:])
        self.tgt_vocab = list(SPECIAL_TOKENS) + sorted(self.tgt_vocab[4:])

        # Build the forward (stoi) and reverse (itos) mapping dictionaries
        for index, token in enumerate(self.src_vocab):
            self.src_stoi[token] = index
            self.src_itos[index] = token

        for index, token in enumerate(self.tgt_vocab):
            self.tgt_stoi[token] = index
            self.tgt_itos[index] = token

        print(f"[dataset] DE vocab size: {len(self.src_vocab)} | EN vocab size: {len(self.tgt_vocab)}")

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        if not self.src_stoi or not self.tgt_stoi:
            raise RuntimeError("Vocabulary is empty. Call build_vocab() first.")
            
        print("[dataset] Converting tokens to index tensors ...")
        self.data = []
        
        # Explicit iteration rather than zip and list comprehensions
        for i in range(len(self.src_tokens)):
            
            # Process Source Sentence
            src_indices = [SOS_IDX]
            for token in self.src_tokens[i]:
                src_indices.append(self.src_stoi.get(token, UNK_IDX))
            src_indices.append(EOS_IDX)
            
            # Process Target Sentence
            tgt_indices = [SOS_IDX]
            for token in self.tgt_tokens[i]:
                tgt_indices.append(self.tgt_stoi.get(token, UNK_IDX))
            tgt_indices.append(EOS_IDX)
            
            # Convert to tensors
            src_tensor = torch.tensor(src_indices, dtype=torch.long)
            tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long)
            
            self.data.append((src_tensor, tgt_tensor))

        print(f"[dataset] {len(self.data)} examples processed and ready.")

    def load_vocab(
        self, 
        src_stoi: Dict[str, int], 
        tgt_stoi: Dict[str, int], 
        src_itos: Dict[int, str], 
        tgt_itos: Dict[int, str],
        src_vocab: List[str], 
        tgt_vocab: List[str]
    ) -> None:
        """
        Injects a pre-built vocabulary from the training set to prevent data leakage
        and ensure consistent integer mapping across validation and test splits.
        """
        self.src_stoi = src_stoi
        self.tgt_stoi = tgt_stoi
        self.src_itos = src_itos
        self.tgt_itos = tgt_itos
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        print(f"[dataset] Vocabulary injected. DE: {len(self.src_vocab)} | EN: {len(self.tgt_vocab)}")

    # ══════════════════════════════════════════════════════════════════════
    # PyTorch Dataset Interface & DataLoader Helpers
    # ══════════════════════════════════════════════════════════════════════

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]

    def _collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pads sequences in a batch to the maximum length of that specific batch.
        """
        src_batch, tgt_batch = zip(*batch)
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
        return src_padded, tgt_padded

    def get_dataloader(self, batch_size: int = 128, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        """
        Builds the DataLoader with the custom collate function.
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )