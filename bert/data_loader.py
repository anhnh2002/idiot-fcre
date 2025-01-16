import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import json

def get_data_loader_BERT(config, data, shuffle = False, drop_last = False, batch_size = None, training: bool = False):
    if batch_size == None:
        batch = min(config.batch_size, len(data))
    else:
        batch = min(batch_size, len(data))
    dataset = BERTDataset(data, config)

    if int(config.imbalanced) and training:
        sampler = ImbalancedBatchSampler([item['relation'] for item in data], batch, config)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch,
            sampler=sampler,
            pin_memory=True,
            num_workers=config.num_workers,
            collate_fn=dataset.collate_fn,
            drop_last=drop_last)
            
    else:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=config.num_workers,
            collate_fn=dataset.collate_fn,
            drop_last=drop_last)

    return data_loader

class BERTDataset(Dataset):    
    def __init__(self, data, config):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], idx)

    def collate_fn(self, data):
        batch_instance = {'ids': [], 'mask': []} 
        batch_label = []
        batch_idx = []

        flag = True

        for item in data:
            if not isinstance(item[0], dict):
                print(item)
                break
            # elif flag:
            #     print(item)
            #     flag = False
            

        batch_label = torch.tensor([item[0]['relation'] for item in data])
        batch_instance['ids'] = torch.tensor([item[0]['ids'] for item in data])
        batch_instance['mask'] = torch.tensor([item[0]['mask'] for item in data])
        batch_idx = torch.tensor([item[1] for item in data])
        
        return batch_instance, batch_label, batch_idx



class ImbalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, config):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.majority_label = int(config.majority_label)
        self.majority_ratio = config.majority_ratio

        
        # Calculate samples per batch for majority and minority classes
        self.majority_samples = int(batch_size * self.majority_ratio)
        self.minority_samples = batch_size - self.majority_samples
        
        # Split indices by label
        self.majority_indices = np.where(self.labels == self.majority_label)[0]
        self.minority_indices = np.where(self.labels != self.majority_label)[0]

        print(f"Labels: {self.labels}")
        print(f"Length Labels: {len(self.labels)}")
        print(f"self.majority_label: {self.majority_label}")
        print(f"self.majority_ratio: {self.majority_ratio}")
        print(f"Length self.majority_indices: {len(self.majority_indices)}")
        print(f"Length self.minority_indices: {len(self.minority_indices)}")

        
    def __iter__(self):
        # Shuffle indices for both groups
        np.random.shuffle(self.majority_indices)
        np.random.shuffle(self.minority_indices)
        
        indices = []  # Store individual indices instead of batches
        majority_idx_pos = 0
        minority_idx_pos = 0
        
        while minority_idx_pos < len(self.minority_indices):
            batch_indices = []
            
            # Get majority samples for this batch
            remaining_majority = len(self.majority_indices) - majority_idx_pos
            maj_samples_to_take = min(self.majority_samples, remaining_majority)
            if maj_samples_to_take > 0:
                majority_idx = self.majority_indices[
                    majority_idx_pos : majority_idx_pos + maj_samples_to_take
                ]
                batch_indices.extend(majority_idx)
                majority_idx_pos += maj_samples_to_take
            
            # Get minority samples for this batch
            remaining_minority = len(self.minority_indices) - minority_idx_pos
            min_samples_to_take = min(self.minority_samples, remaining_minority)
            if min_samples_to_take > 0:
                minority_idx = self.minority_indices[
                    minority_idx_pos : minority_idx_pos + min_samples_to_take
                ]
                batch_indices.extend(minority_idx)
                minority_idx_pos += min_samples_to_take
            
            # Add individual indices to main list
            if batch_indices:
                np.random.shuffle(batch_indices)
                indices.extend(batch_indices)
        
        return iter(indices)  # Return iterator of individual indices
    
    def __len__(self):
        return len(self.minority_indices) // self.minority_samples if len(self.minority_indices) % self.minority_samples == 0 else len(self.minority_indices) // self.minority_samples + 1
    
