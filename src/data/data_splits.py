"""
RiskNICLP-5G Data Splitting Utilities

Provides utilities for creating data splits
"""

import torch
from typing import Tuple, List, Any
from torch.utils.data import Dataset


def create_data_splits(dataset: Dataset, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create train/validation/test splits from dataset with proper 8:1:1 ratio
    
    Args:
        dataset: Input dataset
        train_ratio: Training data ratio (default 0.8)
        val_ratio: Validation data ratio (default 0.1) 
        test_ratio: Test data ratio (default 0.1)
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Validate split ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError("Cannot split empty dataset")
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Calculate split sizes
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size  # Ensure all samples are used
    
    # Generate random indices
    indices = torch.randperm(dataset_size).tolist()
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subset datasets
    train_dataset = SubsetDataset(dataset, train_indices)
    val_dataset = SubsetDataset(dataset, val_indices)
    test_dataset = SubsetDataset(dataset, test_indices)
    
    print(f"Data splits created: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


class SubsetDataset(Dataset):
    """Dataset subset wrapper for supporting index-based data splitting"""
    
    def __init__(self, dataset: Dataset, indices: List[int]):
        """
        Initialize subset dataset
        
        Args:
            dataset: Original dataset
            indices: List of indices to include in subset
        """
        self.dataset = dataset
        self.indices = indices
        
    def __len__(self) -> int:
        """Return subset length"""
        return len(self.indices)
        
    def __getitem__(self, idx: int):
        """Get item from subset by mapping to original dataset index"""
        if idx >= len(self.indices):
            raise IndexError("Subset index out of range")
        original_idx = self.indices[idx]
        return self.dataset[original_idx]


class DataSplitter:
    """Data splitting utility class with stratified sampling support"""
    
    def __init__(self, seed: int = 42):
        """
        Initialize data splitter
        
        Args:
            seed: Random seed for reproducible splits
        """
        self.seed = seed
        torch.manual_seed(seed)
        
    def split_dataset(self, dataset: Dataset, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split dataset into train/val/test using random sampling
        
        Args:
            dataset: Input dataset
            ratios: Tuple of (train_ratio, val_ratio, test_ratio)
            
        Returns:
            Tuple of split datasets
        """
        train_ratio, val_ratio, test_ratio = ratios
        return create_data_splits(dataset, train_ratio, val_ratio, test_ratio, self.seed)
        
    def stratified_split(self, dataset: Dataset, labels: List[int], ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create stratified splits maintaining class balance
        
        Args:
            dataset: Input dataset
            labels: List of labels for stratification
            ratios: Split ratios
            
        Returns:
            Tuple of stratified split datasets
        """
        from sklearn.model_selection import train_test_split
        
        train_ratio, val_ratio, test_ratio = ratios
        
        # Validate ratios
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(ratios)}")
        
        # Create indices array
        indices = list(range(len(dataset)))
        
        # First split: train vs (val + test)
        train_indices, temp_indices, train_labels, temp_labels = train_test_split(
            indices, labels, 
            test_size=(val_ratio + test_ratio),
            random_state=self.seed,
            stratify=labels
        )
        
        # Second split: val vs test
        val_size_ratio = val_ratio / (val_ratio + test_ratio)
        val_indices, test_indices = train_test_split(
            temp_indices, 
            test_size=(1 - val_size_ratio),
            random_state=self.seed,
            stratify=temp_labels
        )
        
        # Create subset datasets
        train_dataset = SubsetDataset(dataset, train_indices)
        val_dataset = SubsetDataset(dataset, val_indices)
        test_dataset = SubsetDataset(dataset, test_indices)
        
        print(f"Stratified splits created: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset 