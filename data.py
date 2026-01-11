import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Tuple, Dict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

class MSADataset(Dataset):
    def __init__(self, sequences: List[Dict], max_length: int = 512,
                 mask_prob: float = 0.15, fixed_seed: int = None):
        self.sequences = sequences
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.mask_token = 23
        self.pad_token = 0
        self.fixed_seed = fixed_seed
        
    def _apply_mlm_masking(self, sequence: np.ndarray, seq_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        masked_sequence = sequence.copy()
        labels = np.full_like(sequence, -100, dtype=np.int32)
        
        if self.fixed_seed is not None: rng = np.random.RandomState(self.fixed_seed + seq_idx)
        else: rng = np.random
        
        for i in range(len(sequence)):
            if rng.random() < self.mask_prob:
                labels[i] = sequence[i]
                
                # BERT-style corruption strategy - prevents naive predictino of mode-label
                rand_val = rng.random()
                if rand_val < 0.8: masked_sequence[i] = self.mask_token
                elif rand_val < 0.9: masked_sequence[i] = rng.randint(1, 22)
                
        return masked_sequence, labels
    
    def _pad_sequence(self, sequence: np.ndarray, is_labels: bool = False) -> np.ndarray:
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        elif len(sequence) < self.max_length:
            # For labels, pad with -100 (ignore index), for input sequences pad with 0
            pad_value = -100 if is_labels else self.pad_token
            padding = np.full(self.max_length - len(sequence), pad_value, dtype=np.int32)
            sequence = np.concatenate([sequence, padding])
        
        return sequence
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        protein_seq = self.sequences[idx]
        sequence = protein_seq['sequence']

        masked_sequence, labels = self._apply_mlm_masking(sequence, idx)

        masked_sequence = self._pad_sequence(masked_sequence, is_labels=False)
        labels = self._pad_sequence(labels, is_labels=True)

        attention_mask = (masked_sequence != self.pad_token).astype(np.float32)

        return {
            'input_ids': torch.tensor(masked_sequence, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
            'labels': torch.tensor(labels, dtype=torch.long),
            'source_file': protein_seq['source_file'],
            'seq_length': protein_seq['length']
        }

def load_protein_data(data_dir: str, max_files: int = 1000,
                     sequences_per_file: int = 10, min_seq_length: int = 20) -> Dict[str, List[Dict]]:
    """ Returns dict mapping filename -> list of dicts (sequence, source_file, seq_idx (in file), length) """
    def _load_single_file(args):
        data_dir, filename, sequences_per_file, min_seq_length = args
        filepath = os.path.join(data_dir, filename)
        
        try:
            data = np.load(filepath)
            sequences = data['sequences']
            residues = data['residues']['res_type']
            
            file_seqs = []
            num_sequences = min(len(sequences), sequences_per_file)
            
            for seq_idx in range(num_sequences):
                seq_info = sequences[seq_idx]
                start_idx = seq_info['res_start']
                end_idx = seq_info['res_end']
                
                if end_idx > start_idx and (end_idx - start_idx) >= min_seq_length:
                    sequence = residues[start_idx:end_idx].astype(np.int32)
                    if len(sequence) > 0:
                        protein_seq = {
                            'sequence': sequence,
                            'source_file': filename,
                            'seq_idx': seq_idx,
                            'length': len(sequence)
                        }
                        file_seqs.append(protein_seq)
            
            return filename, file_seqs
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return filename, []


    files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    files = sorted(files)[:max_files]  # Sort for reproducibility
    
    file_sequences = defaultdict(list)
    args_list = [(data_dir, filename, sequences_per_file, min_seq_length) for filename in files]
    
    max_workers = min(8, multiprocessing.cpu_count())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_load_single_file, args_list)
    
    processed_files = 0
    for filename, file_seqs in results:
        if file_seqs:
            file_sequences[filename] = file_seqs
            processed_files += 1
        
        if processed_files % 100 == 0:
            print(f"Processed {processed_files}/{len(files)} files")
    
    total_sequences = sum(len(seqs) for seqs in file_sequences.values())
    
    return file_sequences

def create_diverse_splits(file_sequences: Dict[str, List[Dict]],
                         train_file_ratio: float = 0.7,
                         val_file_ratio: float = 0.15,
                         max_seqs_per_file_train: int = 5,
                         max_seqs_per_file_val: int = 2) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    all_files = list(file_sequences.keys())
    random.seed(42)  # For reproducible splits
    random.shuffle(all_files)
    
    # Split files
    n_train_files = int(len(all_files) * train_file_ratio)
    n_val_files = int(len(all_files) * val_file_ratio)
    
    train_files = all_files[:n_train_files]
    val_files = all_files[n_train_files:n_train_files + n_val_files]

    test_files = all_files[n_train_files + n_val_files:] # leftovers
    
    print(f"file splits: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    # training set
    train_sequences = []
    for filename in train_files:
        seqs = file_sequences[filename]
        if len(seqs) > max_seqs_per_file_train:
            selected_seqs = random.sample(seqs, max_seqs_per_file_train)
        else:
            selected_seqs = seqs
        
        # All selected sequences go to training
        train_sequences.extend(selected_seqs)
    
    # validation set 
    val_sequences = []
    for filename in val_files:
        seqs = file_sequences[filename]

        num_val_seqs = min(len(seqs), max_seqs_per_file_val)
        if num_val_seqs > 1:
            selected = random.sample(seqs, num_val_seqs)
        else:
            selected = seqs
        val_sequences.extend(selected)
    
    # test set: single sequence per test file
    test_sequences = []
    for filename in test_files:
        seqs = file_sequences[filename]
        if seqs:
            # Take the longest sequence as most representative
            best_seq = max(seqs, key=lambda s: s['length'])
            test_sequences.append(best_seq)
    
    print(f"sequences: {len(train_sequences)} train, {len(val_sequences)} val, "
          f"{len(test_sequences)} test")
    
    return train_sequences, val_sequences, test_sequences

def create_dataloaders(data_dir: str, batch_size: int = 32, max_length: int = 512, 
                       max_files: int = 1000, mask_prob: float = 0.15, sequences_per_file: int = 10) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    create training, validation, and test dataloaders with proper data splitting.
    Returns: (train_loader, val_loader, test_loader)
    """
    
    file_sequences: Dict[str, List[Dict]] = load_protein_data(
        data_dir,
        max_files=max_files,
        sequences_per_file=sequences_per_file
    )
    
    train_sequences, val_sequences, test_sequences = create_diverse_splits(file_sequences) # lists of seqs
    
    train_dataset = MSADataset(train_sequences, max_length=max_length,mask_prob=mask_prob, fixed_seed=42)
    val_dataset = MSADataset(val_sequences, max_length=max_length,mask_prob=mask_prob, fixed_seed=123)
    test_dataset = MSADataset(test_sequences, max_length=max_length,mask_prob=mask_prob, fixed_seed=456)
    
    num_workers = min(8, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"SEQUENC SPLIT: {len(train_sequences)}, {len(val_sequences)}, {len(test_sequences)} ")

    train_files = set(seq['source_file'] for seq in train_sequences)
    val_files = set(seq['source_file'] for seq in val_sequences)
    test_files = set(seq['source_file'] for seq in test_sequences)
    
    print(f"Files: Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    return train_loader, val_loader, test_loader
