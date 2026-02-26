# src/datas/scaling.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os

class DataScaler:
    def __init__(self, method='standard', save_path='artifacts/scaling/'):
        """
        method: 'standard' for StandardScaler or 'minmax' for MinMaxScaler
        save_path: folder to save scaler.pkl
        """
        self.method = method
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.scaler = None

    def fit_scaler(self, X_train: pd.DataFrame):
        """
        Fit scaler only on training data
        """
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("method must be 'standard' or 'minmax'")
        
        self.scaler.fit(X_train)
        print(f"Scaler fitted using {self.method} method.")

    def transform(self, X: pd.DataFrame):
        """
        Transform any dataset using the fitted scaler
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted yet.")
        return pd.DataFrame(self.scaler.transform(X), columns=X.columns)

    def save_scaler(self, filename='scaler.pkl'):
        """
        Save the fitted scaler for later use in FastAPI
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted yet.")
        save_file = os.path.join(self.save_path, filename)
        joblib.dump(self.scaler, save_file)
        print(f"Scaler saved at {save_file}")
        


# ==================================
# Testing block (MUST be outside class)
# ==================================

if __name__ == "__main__":
    print("Testing DataScaler...\n")

    from src.datas.loader import DataLoader
    from sklearn.model_selection import train_test_split

    # Initialize loader
    loader = DataLoader()

    # Load processed dataset
    df = loader.load_csv("dataset/processed/burnout_processed.csv")

    # Separate features and target
    X = df.drop(columns=["burnout_risk"])
    y = df["burnout_risk"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize scaler
    scaler = DataScaler(method="standard")

    # Fit only on train
    scaler.fit_scaler(X_train)

    # Transform both
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    scaler.save_scaler()

    print("\nScaler test successful ✅")