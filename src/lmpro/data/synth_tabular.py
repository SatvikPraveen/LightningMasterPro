# File: src/lmpro/data/synth_tabular.py

"""
Synthetic tabular data generation for regression and classification tasks
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from torch.utils.data import Dataset
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class TabularDatasetConfig:
    """Configuration for synthetic tabular datasets"""
    num_samples: int = 1000
    num_features: int = 20
    num_informative: int = 15
    num_redundant: int = 2
    num_clusters: int = 2
    num_classes: int = 3
    noise_level: float = 0.1
    random_state: int = 42
    save_path: Optional[str] = "data/synthetic/tabular"


class SyntheticTabularDataset(Dataset):
    """Base class for synthetic tabular datasets"""
    
    def __init__(
        self,
        config: TabularDatasetConfig,
        task: str = "classification",
        split: str = "train",
        normalize: bool = True
    ):
        self.config = config
        self.task = task
        self.split = split
        self.normalize = normalize
        
        # Generate data
        self.X, self.y = self._generate_data()
        
        # Apply normalization
        if self.normalize:
            self.scaler = StandardScaler()
            if split == "train":
                self.X = self.scaler.fit_transform(self.X)
            else:
                # In practice, you'd save the scaler from training
                self.scaler.fit(self.X)  # Simplified for synthetic data
                self.X = self.scaler.transform(self.X)
        
        # Convert to tensors
        self.X_tensor = torch.tensor(self.X, dtype=torch.float32)
        self.y_tensor = torch.tensor(self.y, dtype=torch.long if task == "classification" else torch.float32)
        
        # Feature names
        self.feature_names = [f"feature_{i}" for i in range(self.X.shape[1])]
        
    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data based on task type"""
        if self.task == "classification":
            return self._generate_classification_data()
        elif self.task == "regression":
            return self._generate_regression_data()
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def _generate_classification_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic classification data"""
        X, y = make_classification(
            n_samples=self.config.num_samples,
            n_features=self.config.num_features,
            n_informative=self.config.num_informative,
            n_redundant=self.config.num_redundant,
            n_clusters_per_class=self.config.num_clusters,
            n_classes=self.config.num_classes,
            class_sep=1.0,
            flip_y=self.config.noise_level,
            random_state=self.config.random_state + (0 if self.split == "train" else 1)
        )
        
        return X, y
    
    def _generate_regression_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic regression data"""
        X, y = make_regression(
            n_samples=self.config.num_samples,
            n_features=self.config.num_features,
            n_informative=self.config.num_informative,
            n_targets=1,
            noise=self.config.noise_level * 10,
            random_state=self.config.random_state + (0 if self.split == "train" else 1)
        )
        
        return X, y.flatten()
    
    def __len__(self) -> int:
        return len(self.X_tensor)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X_tensor[idx], self.y_tensor[idx]
    
    def get_pandas_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for analysis"""
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df['target'] = self.y
        return df


class ComplexTabularDataset(Dataset):
    """More complex tabular dataset with mixed data types"""
    
    def __init__(
        self,
        config: TabularDatasetConfig,
        task: str = "classification",
        split: str = "train",
        include_categorical: bool = True,
        include_missing: bool = True
    ):
        self.config = config
        self.task = task
        self.split = split
        self.include_categorical = include_categorical
        self.include_missing = include_missing
        
        # Generate complex data
        self.df = self._generate_complex_data()
        
        # Prepare features and targets
        self.X, self.y = self._prepare_features_targets()
        
        # Convert to tensors
        self.X_tensor = torch.tensor(self.X.values, dtype=torch.float32)
        self.y_tensor = torch.tensor(self.y.values, dtype=torch.long if task == "classification" else torch.float32)
    
    def _generate_complex_data(self) -> pd.DataFrame:
        """Generate complex tabular data with different column types"""
        np.random.seed(self.config.random_state + (0 if self.split == "train" else 1))
        
        data = {}
        
        # Numerical features
        num_numerical = self.config.num_features // 2
        for i in range(num_numerical):
            if i % 3 == 0:  # Normal distribution
                data[f'num_{i}'] = np.random.normal(0, 1, self.config.num_samples)
            elif i % 3 == 1:  # Uniform distribution
                data[f'num_{i}'] = np.random.uniform(-2, 2, self.config.num_samples)
            else:  # Exponential distribution
                data[f'num_{i}'] = np.random.exponential(1, self.config.num_samples)
        
        # Categorical features
        if self.include_categorical:
            num_categorical = self.config.num_features - num_numerical
            categories = {
                'category_A': ['cat1', 'cat2', 'cat3', 'cat4'],
                'category_B': ['type_x', 'type_y', 'type_z'],
                'category_C': ['low', 'medium', 'high'],
                'category_D': ['A', 'B', 'C', 'D', 'E']
            }
            
            cat_names = list(categories.keys())
            for i in range(num_categorical):
                cat_name = cat_names[i % len(cat_names)]
                cat_values = categories[cat_name]
                data[f'cat_{i}'] = np.random.choice(cat_values, self.config.num_samples)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Generate target variable
        if self.task == "classification":
            # Create target based on some features
            target_score = 0
            for col in df.select_dtypes(include=[np.number]).columns[:3]:
                target_score += df[col].values
            
            # Convert to classes
            percentiles = np.percentile(target_score, [33, 67])
            target = np.digitize(target_score, percentiles)
            df['target'] = target
        else:  # regression
            # Create continuous target
            target = 0
            for col in df.select_dtypes(include=[np.number]).columns[:5]:
                target += df[col].values * np.random.uniform(-2, 2)
            target += np.random.normal(0, self.config.noise_level, len(target))
            df['target'] = target
        
        # Add missing values
        if self.include_missing:
            missing_cols = np.random.choice(df.columns[:-1], size=min(3, len(df.columns)-1), replace=False)
            for col in missing_cols:
                missing_idx = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
                df.loc[missing_idx, col] = np.nan
        
        return df
    
    def _prepare_features_targets(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and targets for training"""
        # Separate features and target
        X = self.df.drop('target', axis=1)
        y = self.df['target']
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Normalize numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        
        return X, y
    
    def __len__(self) -> int:
        return len(self.X_tensor)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X_tensor[idx], self.y_tensor[idx]
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return list(self.X.columns)


class TimeVaryingTabularDataset(Dataset):
    """Tabular dataset with time-varying features"""
    
    def __init__(
        self,
        config: TabularDatasetConfig,
        sequence_length: int = 10,
        task: str = "classification",
        split: str = "train"
    ):
        self.config = config
        self.sequence_length = sequence_length
        self.task = task
        self.split = split
        
        # Generate time series data
        self.sequences, self.targets = self._generate_time_varying_data()
        
        # Convert to tensors
        self.sequences_tensor = torch.tensor(self.sequences, dtype=torch.float32)
        self.targets_tensor = torch.tensor(self.targets, dtype=torch.long if task == "classification" else torch.float32)
    
    def _generate_time_varying_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate time-varying tabular data"""
        np.random.seed(self.config.random_state + (0 if self.split == "train" else 1))
        
        sequences = []
        targets = []
        
        for _ in range(self.config.num_samples):
            # Generate a sequence
            sequence = np.zeros((self.sequence_length, self.config.num_features))
            
            # Add trends and patterns
            for t in range(self.sequence_length):
                for f in range(self.config.num_features):
                    if f % 4 == 0:  # Linear trend
                        sequence[t, f] = t * 0.1 + np.random.normal(0, 0.1)
                    elif f % 4 == 1:  # Seasonal pattern
                        sequence[t, f] = np.sin(2 * np.pi * t / 5) + np.random.normal(0, 0.1)
                    elif f % 4 == 2:  # Random walk
                        if t == 0:
                            sequence[t, f] = np.random.normal(0, 1)
                        else:
                            sequence[t, f] = sequence[t-1, f] + np.random.normal(0, 0.1)
                    else:  # Random noise
                        sequence[t, f] = np.random.normal(0, 1)
            
            sequences.append(sequence)
            
            # Generate target based on sequence statistics
            if self.task == "classification":
                # Use mean and trend as features for classification
                mean_values = np.mean(sequence, axis=0)
                trend_values = sequence[-1] - sequence[0]
                target_score = np.sum(mean_values[:3]) + np.sum(trend_values[:3])
                target = 0 if target_score < -0.5 else 1 if target_score < 0.5 else 2
            else:  # regression
                # Use sequence statistics for regression
                target = np.sum(np.mean(sequence, axis=0)) + np.random.normal(0, 0.1)
            
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def __len__(self) -> int:
        return len(self.sequences_tensor)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences_tensor[idx], self.targets_tensor[idx]


def create_synthetic_tabular_dataset(
    config: TabularDatasetConfig,
    task: str = "classification",
    splits: List[str] = ["train", "val", "test"],
    split_ratios: List[float] = [0.7, 0.15, 0.15],
    dataset_type: str = "simple"
) -> dict:
    """Create synthetic tabular datasets"""
    datasets = {}
    
    total_samples = config.num_samples
    split_sizes = [int(ratio * total_samples) for ratio in split_ratios]
    
    for split, size in zip(splits, split_sizes):
        split_config = TabularDatasetConfig(
            num_samples=size,
            num_features=config.num_features,
            num_informative=config.num_informative,
            num_redundant=config.num_redundant,
            num_clusters=config.num_clusters,
            num_classes=config.num_classes,
            noise_level=config.noise_level,
            random_state=config.random_state,
            save_path=config.save_path
        )
        
        if dataset_type == "simple":
            datasets[split] = SyntheticTabularDataset(split_config, task=task, split=split)
        elif dataset_type == "complex":
            datasets[split] = ComplexTabularDataset(split_config, task=task, split=split)
        elif dataset_type == "time_varying":
            datasets[split] = TimeVaryingTabularDataset(split_config, task=task, split=split)
    
    return datasets


def create_synthetic_regression_dataset(
    config: TabularDatasetConfig,
    splits: List[str] = ["train", "val", "test"],
    split_ratios: List[float] = [0.7, 0.15, 0.15]
) -> dict:
    """Create synthetic regression datasets"""
    return create_synthetic_tabular_dataset(config, task="regression", splits=splits, split_ratios=split_ratios)


def create_synthetic_classification_dataset(
    config: TabularDatasetConfig,
    splits: List[str] = ["train", "val", "test"],
    split_ratios: List[float] = [0.7, 0.15, 0.15]
) -> dict:
    """Create synthetic classification datasets"""
    return create_synthetic_tabular_dataset(config, task="classification", splits=splits, split_ratios=split_ratios)


def visualize_tabular_data(dataset: Dataset, save_path: Optional[str] = None) -> None:
    """Visualize tabular dataset"""
    # Get data as DataFrame if possible
    if hasattr(dataset, 'get_pandas_dataframe'):
        df = dataset.get_pandas_dataframe()
    else:
        # Convert from tensor data
        X = dataset.X_tensor.numpy()
        y = dataset.y_tensor.numpy()
        feature_names = getattr(dataset, 'feature_names', [f'feature_{i}' for i in range(X.shape[1])])
        
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Correlation heatmap
    numerical_cols = df.select_dtypes(include=[np.number]).columns[:10]  # Limit to first 10 for readability
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 0])
    axes[0, 0].set_title('Feature Correlation Matrix')
    
    # Target distribution
    if dataset.task == "classification":
        df['target'].value_counts().plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Class Distribution')
    else:
        df['target'].hist(bins=30, ax=axes[0, 1])
        axes[0, 1].set_title('Target Distribution')
    
    # Feature distributions (first few features)
    features_to_plot = numerical_cols[:4]
    for i, feature in enumerate(features_to_plot):
        if i < 2:
            row, col = 1, i
            df[feature].hist(bins=30, ax=axes[row, col], alpha=0.7)
            axes[row, col].set_title(f'{feature} Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Tabular data visualization saved to {save_path}")
    
    plt.show()


def analyze_dataset_quality(dataset: Dataset) -> Dict[str, float]:
    """Analyze the quality of the synthetic dataset"""
    if hasattr(dataset, 'get_pandas_dataframe'):
        df = dataset.get_pandas_dataframe()
    else:
        X = dataset.X_tensor.numpy()
        y = dataset.y_tensor.numpy()
        feature_names = getattr(dataset, 'feature_names', [f'feature_{i}' for i in range(X.shape[1])])
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
    
    analysis = {}
    
    # Basic statistics
    analysis['num_samples'] = len(df)
    analysis['num_features'] = len(df.columns) - 1
    analysis['missing_percentage'] = (df.isnull().sum().sum() / df.size) * 100
    
    # Feature correlations
    numerical_features = df.select_dtypes(include=[np.number]).columns
    if len(numerical_features) > 1:
        corr_matrix = df[numerical_features].corr().abs()
        # Average correlation (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        analysis['avg_feature_correlation'] = corr_matrix.values[mask].mean()
    
    # Target analysis
    if hasattr(dataset, 'task'):
        if dataset.task == "classification":
            analysis['class_balance'] = df['target'].value_counts().std() / df['target'].value_counts().mean()
            analysis['num_classes'] = df['target'].nunique()
        else:
            analysis['target_std'] = df['target'].std()
            analysis['target_range'] = df['target'].max() - df['target'].min()
    
    return analysis


def print_dataset_summary(dataset: Dataset, name: str = "Dataset") -> None:
    """Print summary statistics for the dataset"""
    print(f"\n{name} Summary:")
    print(f"Length: {len(dataset)}")
    
    analysis = analyze_dataset_quality(dataset)
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")