"""
Data Loader Module
------------------
Simple and professional DataLoader
with tqdm progress support.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class DataLoader:
    """
    Utility class to handle dataset operations.
    """

    def __init__(self, base_dir: str = None, verbose: bool = True):
        self.base_dir = base_dir if base_dir else os.getcwd()
        self.verbose = verbose

    def _get_full_path(self, path: str) -> str:
        """
        Convert relative path to full path.
        """
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_dir, path)

    def load_csv(self, path: str, chunksize: int = None) -> pd.DataFrame:
        """
        Load CSV file.
        If chunksize is given, it loads file in parts and shows progress bar.
        """

        full_path = self._get_full_path(path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")

        # Normal loading
        if chunksize is None:
            df = pd.read_csv(full_path)

        # Chunk loading (for large files)
        else:
            chunks = []
            reader = pd.read_csv(full_path, chunksize=chunksize)

            print("Loading data with progress bar...")

            for chunk in tqdm(reader, desc="Reading CSV"):
                chunks.append(chunk)

            df = pd.concat(chunks, ignore_index=True)

        if self.verbose:
            print("Dataset loaded successfully.")
            print(f"Shape: {df.shape}")

        return df

    def save_csv(self, df: pd.DataFrame, path: str) -> None:
        """
        Save DataFrame to CSV.
        """
        full_path = self._get_full_path(path)

        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        df.to_csv(full_path, index=False)

        if self.verbose:
            print(f"Dataset saved to: {full_path}")

    def sample_data(self, df: pd.DataFrame, fraction: float = 0.1) -> pd.DataFrame:
        """
        Take random sample from dataset.
        """
        sampled_df = df.sample(frac=fraction, random_state=42)

        if self.verbose:
            print(f"Sampled {fraction * 100}% of data.")

        return sampled_df

    def split_data(self, df: pd.DataFrame, target_column: str,
                   test_size: float = 0.2):
        """
        Split dataset into train and test sets.
        """

        if target_column not in df.columns:
            raise ValueError(f"{target_column} not found in dataset.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y  # keeps class balance
        )

        if self.verbose:
            print("Data split completed.")
            print(f"Train shape: {X_train.shape}")
            print(f"Test shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test


# ==================================
# Testing block (MUST be outside class)
# ==================================

if __name__ == "__main__":
    print("Testing DataLoader...\n")

    loader = DataLoader()

    # Load dataset
    df = loader.load_csv("dataset/raw/wfh_burnout_dataset.csv")

    # Take sample
    sample_df = loader.sample_data(df, 0.1)

    # Save sample
    loader.save_csv(sample_df, "dataset/processed/sample_test.csv")

    # Split data
    X_train, X_test, y_train, y_test = loader.split_data(df, "burnout_risk")

    print("\nEverything is working correctly!")