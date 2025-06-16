# Classes and functions for data processing and model training

import pandas as pd
import numpy as np
import torch
import datetime as datetime
import itertools
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score, r2_score, precision_score, recall_score, f1_score, median_absolute_error
from tqdm.notebook import tqdm
import gc
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=25):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                         (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class VehicleTestDataset(Dataset):
    def __init__(self, sequences, attention_masks, targets_mileage, targets_scrapped):
        # Ensure all inputs are float32 arrays
        self.sequences = torch.from_numpy(np.array(sequences, dtype=np.float32))
        self.attention_masks = torch.from_numpy(np.array(attention_masks, dtype=np.float32))
        self.targets_mileage = torch.from_numpy(np.array(targets_mileage, dtype=np.float32).reshape(-1, 1))
        self.targets_scrapped = torch.from_numpy(np.array(targets_scrapped, dtype=np.float32).reshape(-1, 1))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (self.sequences[idx], 
                self.attention_masks[idx],
                self.targets_mileage[idx],
                self.targets_scrapped[idx])

class VehicleTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.mileage_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(dim_feedforward, 1)
        )
        
        self.scrapped_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_feedforward, 1),  # Output single value
            nn.Sigmoid()
        )

    # def forward(self, src, src_key_padding_mask):
    #     # Add dimension checks
    #     batch_size, seq_len, feat_dim = src.shape
    #     if src_key_padding_mask.shape != (batch_size, seq_len):
    #         raise ValueError(f"Mask shape {src_key_padding_mask.shape} doesn't match input shape {src.shape}")
        
    #     x = self.embedding(src)
    #     x = self.pos_encoder(x)
        
    #     # Invert the mask for transformer (True means ignore)
    #     transformer_mask = ~src_key_padding_mask.bool()
        
    #     output = self.transformer_encoder(
    #         x,
    #         src_key_padding_mask=transformer_mask
    #     )
        
    #     # Get the last non-padded element for each sequence
    #     last_real_idx = (~transformer_mask).sum(dim=1) - 1
    #     last_hidden = output[torch.arange(batch_size), last_real_idx]
        
    #     mileage_pred = self.mileage_head(last_hidden)
    #     scrapped_pred = self.scrapped_head(last_hidden)
        
    #     # Ensure outputs have correct shapes
    #     mileage_pred = self.mileage_head(last_hidden)  # Shape: [batch_size, 1]
    #     scrapped_pred = self.scrapped_head(last_hidden)  # Shape: [batch_size, 1]
        
    #     return mileage_pred, scrapped_pred
    
    def forward(self, src, src_key_padding_mask):
        # Input validation
        if torch.isnan(src).any():
            print("NaN detected in input")
            
        # Embedding layer
        x = self.embedding(src)
        if torch.isnan(x).any():
            print("NaN detected after embedding")
            
        # Positional encoding
        x = self.pos_encoder(x)
        if torch.isnan(x).any():
            print("NaN detected after positional encoding")
        
        # Transformer mask
        transformer_mask = ~src_key_padding_mask.bool()
        
        # Transformer
        output = self.transformer_encoder(x, src_key_padding_mask=transformer_mask)
        if torch.isnan(output).any():
            print("NaN detected after transformer encoder")
        
        # Get last non-padded element
        last_real_idx = transformer_mask.sum(dim=1) - 1
        batch_indices = torch.arange(len(src), device=src.device)
        last_hidden = output[batch_indices, last_real_idx]
        
        if torch.isnan(last_hidden).any():
            print("NaN detected in last hidden state")
        
        # Predictions with gradient clipping
        mileage_pred = torch.clip(self.mileage_head(last_hidden), -1e6, 1e6)
        if torch.isnan(mileage_pred).any():
            print("NaN detected in mileage prediction")
            
        scrapped_pred = torch.clip(self.scrapped_head(last_hidden), 0, 1)
        if torch.isnan(scrapped_pred).any():
            print("NaN detected in scrapped prediction")
        
        return mileage_pred, scrapped_pred
    
def pad_sequences(sequences, max_len=8):
    """Pad sequences and create attention masks"""
    padded_sequences = []
    attention_masks = []
    
    for seq in sequences:
        # Convert sequence to float32 array if it isn't already
        seq = np.array(seq, dtype=np.float32)
        
        if len(seq) > max_len:
            seq = seq[-max_len:]
            
        pad_length = max_len - len(seq)
        
        # Ensure padding maintains float32 dtype
        padded_seq = np.pad(
            seq,
            ((0, pad_length), (0, 0)),
            mode='constant',
            constant_values=0.0
        ).astype(np.float32)
        
        mask = np.ones(len(seq))
        mask = np.pad(
            mask,
            (0, pad_length),
            mode='constant',
            constant_values=0.0
        ).astype(np.float32)
        
        padded_sequences.append(padded_seq)
        attention_masks.append(mask)
    
    # Stack arrays instead of using array()
    return np.stack(padded_sequences), np.stack(attention_masks)

def create_training_sequences(df, scaler, min_sequence_length=3, max_sequence_length=8, batch_size=25_000):
    """
    Create sequences efficiently by processing data in batches with vectorized operations
    
    A sequence is at least three tests plus the next test to predict
    Therefore the minimum number of tests to create a sequence is 4
    
    """
    sequences = []
    target_mileages = []
    target_scrapped = []
    next_ages = []
    
    # Pre-calculate column indices for performance
    cols_to_drop = ['vehicle_id']
    feature_cols = [col for col in df.columns if col not in cols_to_drop]
    
    # Process in batches using groupby
    for _, group in tqdm(df.groupby('vehicle_id'), desc="Creating Sequences"):
        
        # Only keep at least 4 tests for the minimum sequence length of 3
        if len(group) < min_sequence_length + 1:
            continue
            
        # Convert to numpy array once for the group
        group_array = group[feature_cols].values
        
        # Generate sequences using array slicing
        for i in range(min_sequence_length, len(group_array)-1):
            
            seq = group_array[max(0, i-max_sequence_length+1):i+1]
            
            # Get next values using direct indexing
            next_values = group_array[i+1]
            next_mileage = next_values[feature_cols.index('mileage_per_year')]
            next_scrapped = next_values[feature_cols.index('last_test')]
            next_age = next_values[feature_cols.index('age_year')]
            
            sequences.append(seq)
            target_mileages.append(next_mileage)
            target_scrapped.append(next_scrapped)
            next_ages.append(next_age)
            
            # Batch processing
            if len(sequences) >= batch_size:
                yield sequences, target_mileages, target_scrapped
                sequences, target_mileages, target_scrapped = [], [], []

    # Print statistics after processing
    if sequences:  # For remaining sequences
        yield sequences, target_mileages, target_scrapped
        
    print(f"\nSequence Statistics:")
    print(f"Feature columns: {feature_cols}")
    print(f"Average sequence length: {np.mean([len(s) for s in sequences]):.2f}")
    
    # Get scaler parameters for mileage_between_tests (first feature)
    mileage_mean = scaler.mean_[0]
    mileage_scale = scaler.scale_[0]
    
    # Get scaler parameters for age_year (index 6 in scaled features)
    age_mean = scaler.mean_[2]  # age_year is the 3rd numerical column
    age_scale = scaler.scale_[2]  # age_year is the 3rd numerical column
    
    # Denormalize sequence values for debugging
    print(f"Denormalized sequence values:")
    denorm_mileage = [m * mileage_scale + mileage_mean for m in target_mileages]
    denorm_age = [a * age_scale + age_mean for a in next_ages]

    mean_denorm_mileage = np.mean(denorm_mileage)
    mean_denorm_age = np.mean(denorm_age)
    
    print(f"Mean denormalized mileage: {mean_denorm_mileage:.2f}")
    print(f"Mean denormalized age: {mean_denorm_age:.2f}")
    
    denorm_age_rounded = [round(a) for a in denorm_age]
    age_counts = pd.Series(denorm_age_rounded).value_counts().sort_index()
    print("Age distribution after denormalization:")
    print(age_counts)
    
    # Finally print the number of sequences
    print(f"Total sequences: {len(sequences)}")
    
def prepare_training_data(data, test_size=0.2, random_state=42, batch_size=25_000):
    """Prepare data in batches to manage memory"""
    print("Starting data preparation...")
    print("Preprocessing data...")
    
    # Only keep the columns we need for training
    categorical_cols = ['fuel_type', 'last_test']
    numerical_cols = ['mileage_per_year', 'test_mileage', 'age_year', 'time_between_tests']
    training_cols = ['vehicle_id'] + categorical_cols + numerical_cols
    
    df = data[training_cols].copy()
    
    # Split vehicle IDs first
    vehicle_ids = df['vehicle_id'].unique()
    n_test = int(len(vehicle_ids) * test_size)
    np.random.seed(random_state)
    test_vehicle_ids = np.random.choice(vehicle_ids, size=n_test, replace=False)
    
    train_df = df[~df['vehicle_id'].isin(test_vehicle_ids)]
    test_df = df[df['vehicle_id'].isin(test_vehicle_ids)]
    
    # Fit preprocessing only on training data
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        # Fit on training data only
        label_encoders[col].fit(train_df[col])
        # Transform both train and test
        train_df[col] = label_encoders[col].transform(train_df[col])
        test_df[col] = label_encoders[col].transform(test_df[col])
    
    # Scale numerical features using only training data
    scaler = StandardScaler()
    scaler.fit(train_df[numerical_cols])
    train_df[numerical_cols] = scaler.transform(train_df[numerical_cols])
    test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])
    
    # Process training data in batches
    train_sequences = []
    train_masks = []
    train_mileages = []
    train_scrapped = []
    
    for batch_sequences, batch_mileages, batch_scrapped in create_training_sequences(train_df, scaler,  batch_size=batch_size):
        # Convert sequences to float32 before padding
        batch_sequences = [np.array(seq, dtype=np.float32) for seq in batch_sequences]
        padded_seqs, masks = pad_sequences(batch_sequences)
        
        train_sequences.append(padded_seqs)
        train_masks.append(masks)
        train_mileages.extend(batch_mileages)
        train_scrapped.extend(batch_scrapped)
        
        # Clear memory
        gc.collect()
    
    # Combine batches using np.vstack instead of np.concatenate
    train_sequences = np.vstack(train_sequences)
    train_masks = np.vstack(train_masks)
    train_mileages = np.array(train_mileages, dtype=np.float32)
    train_scrapped = np.array(train_scrapped, dtype=np.float32)
    
    # Process test data similarly
    test_sequences = []
    test_masks = []
    test_mileages = []
    test_scrapped = []
    
    for batch_sequences, batch_mileages, batch_scrapped in create_training_sequences(test_df, scaler,  batch_size=batch_size):
        batch_sequences = [np.array(seq, dtype=np.float32) for seq in batch_sequences]
        padded_seqs, masks = pad_sequences(batch_sequences)
        
        test_sequences.append(padded_seqs)
        test_masks.append(masks)
        test_mileages.extend(batch_mileages)
        test_scrapped.extend(batch_scrapped)
        
        # Clear memory
        gc.collect()
    
    # Combine batches
    test_sequences = np.vstack(test_sequences)
    test_masks = np.vstack(test_masks)
    test_mileages = np.array(test_mileages, dtype=np.float32)
    test_scrapped = np.array(test_scrapped, dtype=np.float32)
    
    # Create datasets
    train_dataset = VehicleTestDataset(
        train_sequences, train_masks, train_mileages, train_scrapped
    )
    test_dataset = VehicleTestDataset(
        test_sequences, test_masks, test_mileages, test_scrapped
    )
    
    return train_dataset, test_dataset, label_encoders, scaler

def calculate_class_weights(train_loader, device):
    """Calculate class weights based on class distribution"""
    print("Calculating class weights...")
    total_samples = 0
    scrapped_count = 0
    
    for _, _, _, batch_scrapped in tqdm(train_loader):
        batch_scrapped = batch_scrapped.to(device)
        total_samples += batch_scrapped.size(0)
        scrapped_count += batch_scrapped.sum().item()
    
    not_scrapped_count = total_samples - scrapped_count
    
    # Simple inverse of class frequencies
    weights = {
        0: 1 / not_scrapped_count,
        1: 1 / scrapped_count
    }

    # Normalize weights
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    
    print(f"Class distribution:")
    print(f"Not scrapped: {not_scrapped_count:,} ({not_scrapped_count/total_samples:.1%})")
    print(f"Scrapped: {scrapped_count:,} ({scrapped_count/total_samples:.1%})")
    print(f"Calculated weights: {weights}")
    
    return weights

def find_optimal_threshold(model, val_loader, device, n_thresholds=100):
    """Find optimal threshold using validation data"""
    print("Finding optimal threshold...")
    true_labels = []
    pred_probs = []
    
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Collecting predictions"):
            sequences, masks, _, batch_scrapped = [x.to(device) for x in batch_data]
            _, scrapped_pred = model(sequences, masks)
            
            pred_probs.extend(scrapped_pred.cpu().numpy())
            true_labels.extend(batch_scrapped.cpu().numpy())
    
    pred_probs = np.array(pred_probs)
    true_labels = np.array(true_labels)
    
    # Test different thresholds
    thresholds = np.linspace(0.1, 0.9, n_thresholds)
    results = []
    
    for threshold in tqdm(thresholds, desc="Testing thresholds"):
        pred_labels = (pred_probs > threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        try:
            auc_roc = roc_auc_score(true_labels, pred_probs)
        except:
            auc_roc = 0
            
        precision = precision_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)
        
        predicted_scrapped = np.sum(pred_labels)
        actual_scrapped = np.sum(true_labels)
        scrapped_ratio = predicted_scrapped / actual_scrapped if actual_scrapped > 0 else 0
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predicted_scrapped': predicted_scrapped,
            'actual_scrapped': actual_scrapped,
            'scrapped_ratio': scrapped_ratio
        })
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Find best threshold based on F1 score
    best_idx = results_df['f1'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    
    print("\nBest threshold results:")
    print(f"Threshold: {best_threshold:.3f}")
    print(f"Accuracy: {results_df.loc[best_idx, 'accuracy']:.3f}")
    print(f"AUC-ROC: {results_df.loc[best_idx, 'auc_roc']:.3f}")
    print(f"Precision: {results_df.loc[best_idx, 'precision']:.3f}")
    print(f"Recall: {results_df.loc[best_idx, 'recall']:.3f}")
    print(f"F1 Score: {results_df.loc[best_idx, 'f1']:.3f}")
    print(f"Predicted/Actual Scrapped Ratio: {results_df.loc[best_idx, 'scrapped_ratio']:.3f}")
    
    return best_threshold, results_df

def train_model(model, train_loader, val_loader, num_epochs, device, lr=0.001):
    """Training with weighted loss for imbalanced classes"""
    print(f"Starting training on device: {device}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_loader, device)
    
    # Loss functions
    mileage_criterion = nn.MSELoss()
    scrapped_criterion = nn.BCELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    # Initialize metrics tracking
    train_metrics = {
        'mileage_losses': [],
        'scrapped_losses': [],
        'total_losses': []
    }
    val_metrics = {
        'mileage_losses': [],
        'scrapped_losses': [],
        'rmse': [],
        'auc': [],
        'accuracy': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_mileage_loss = 0
        train_scrapped_loss = 0
        batch_count = 0
        
        train_progress = tqdm(train_loader, desc="Training", smoothing=1.0)
        for batch_data in train_progress:
            sequences, masks, batch_mileage, batch_scrapped = [
                x.to(device) for x in batch_data
            ]
            
            # Add shape validation
            if batch_mileage.shape != batch_scrapped.shape:
                print(f"Shape mismatch - Mileage: {batch_mileage.shape}, Scrapped: {batch_scrapped.shape}")
                continue
                
            # Ensure proper shapes and types
            batch_mileage = batch_mileage.float().view(-1, 1)
            batch_scrapped = batch_scrapped.float().view(-1, 1)
            
            optimizer.zero_grad()
            
            mileage_pred, scrapped_pred = model(sequences, masks)
            
            # Validate predictions
            if torch.isnan(mileage_pred).any() or torch.isnan(scrapped_pred).any():
                print("Warning: NaN values in predictions")
                continue
                
            # Ensure scrapped predictions are between 0 and 1
            scrapped_pred = torch.clamp(scrapped_pred, 0, 1)
            
            # Calculate losses with validation
            try:
                mileage_loss = mileage_criterion(mileage_pred, batch_mileage)
                scrapped_loss = scrapped_criterion(scrapped_pred, batch_scrapped)
            except RuntimeError as e:
                print(f"Loss calculation error: {e}")
                print(f"Pred shape: {scrapped_pred.shape}, Target shape: {batch_scrapped.shape}")
                print(f"Pred range: [{scrapped_pred.min()}, {scrapped_pred.max()}]")
                print(f"Target range: [{batch_scrapped.min()}, {batch_scrapped.max()}]")
                continue
            
            # Apply class weights manually if needed
            if class_weights is not None:
                weight = torch.where(batch_scrapped > 0.5, 
                                   torch.tensor(class_weights[1]).to(device),
                                   torch.tensor(class_weights[0]).to(device))
                scrapped_loss = (scrapped_loss * weight).mean()
            
            # Combine losses
            total_loss = mileage_loss + 2.0 * scrapped_loss
            
            # Check for invalid loss values
            if not torch.isfinite(total_loss):
                print(f"Warning: Invalid loss value: {total_loss.item()}")
                continue
                
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_mileage_loss += mileage_loss.item()
            train_scrapped_loss += scrapped_loss.item()
            batch_count += 1
        
        # Validation phase
        model.eval()
        val_mileage_loss = 0
        val_scrapped_loss = 0
        val_mileage_preds = []
        val_scrapped_preds = []
        val_mileage_true = []
        val_scrapped_true = []
        val_batch_count = 0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc="Validation")
            for batch_data in val_progress:
                sequences, masks, batch_mileage, batch_scrapped = [
                    x.to(device) for x in batch_data
                ]
                
                mileage_pred, scrapped_pred = model(sequences, masks)
                
                mileage_loss = mileage_criterion(mileage_pred, batch_mileage)
                scrapped_loss = scrapped_criterion(scrapped_pred, batch_scrapped)
                
                val_mileage_loss += mileage_loss.item()
                val_scrapped_loss += scrapped_loss.item()
                val_batch_count += 1
                
                val_mileage_preds.extend(mileage_pred.cpu().numpy())
                val_scrapped_preds.extend(scrapped_pred.cpu().numpy())
                val_mileage_true.extend(batch_mileage.cpu().numpy())
                val_scrapped_true.extend(batch_scrapped.cpu().numpy())
        
        # Calculate metrics
        train_mileage_loss /= batch_count
        train_scrapped_loss /= batch_count
        val_mileage_loss /= val_batch_count
        val_scrapped_loss /= val_batch_count
        
        val_mileage_preds = np.array(val_mileage_preds)
        val_mileage_true = np.array(val_mileage_true)
        
        rmse = np.sqrt(mean_squared_error(val_mileage_true, val_mileage_preds))
        auc = roc_auc_score(val_scrapped_true, val_scrapped_preds)
        accuracy = accuracy_score(
            val_scrapped_true,
            [1 if p > 0.5 else 0 for p in val_scrapped_preds]
        )
        
        # Store metrics
        train_metrics['mileage_losses'].append(train_mileage_loss)
        train_metrics['scrapped_losses'].append(train_scrapped_loss)
        train_metrics['total_losses'].append(train_mileage_loss + train_scrapped_loss)
        
        val_metrics['mileage_losses'].append(val_mileage_loss)
        val_metrics['scrapped_losses'].append(val_scrapped_loss)
        val_metrics['rmse'].append(rmse)
        val_metrics['auc'].append(auc)
        val_metrics['accuracy'].append(accuracy)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Training Metrics:")
        print(f"  Mileage Loss: {train_mileage_loss:.4f}")
        print(f"  Scrapped Loss: {train_scrapped_loss:.4f}")
        print(f"Validation Metrics:")
        print(f"  Mileage Loss: {val_mileage_loss:.4f}")
        print(f"  Scrapped Loss: {val_scrapped_loss:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        
        # Save best model
        val_loss = val_mileage_loss + val_scrapped_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved! (Epoch {best_epoch})")
        
        # Update learning rate
        scheduler.step(val_loss)
    
    print("\nTraining completed!")
    print(f"Best model was saved at epoch {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Find optimal threshold
    best_threshold, threshold_results = find_optimal_threshold(model, val_loader, device)
    
    return train_metrics, val_metrics, best_threshold, threshold_results

def hyperparameter_tuning(data, param_grid, device, train_loader, test_loader, scaler):
    """
    Perform grid search hyperparameter tuning for the transformer model.
    
    Args:
        data: Input dataset
        param_grid: Dictionary of parameters to tune with lists of values
        base_batch_size: Base batch size for training
        test_size: Proportion of data to use for testing
    
    Returns:
        DataFrame containing results for all parameter combinations
    """
    results = []
    
    # Generate all possible combinations of parameters
    param_names = list(param_grid.keys())
    param_values = [param_grid[param] for param in param_names]
    param_combinations = list(itertools.product(*param_values))
    
    total_combinations = len(param_combinations)
    print(f"Total parameter combinations to try: {total_combinations}")
    
    for i, params in enumerate(param_combinations, 1):
        current_params = dict(zip(param_names, params))
        print(f"\nTrying combination {i}/{total_combinations}:")
        for name, value in current_params.items():
            print(f"{name}: {value}")
            
        try:
  
            # Initialize model with current parameters
            model = VehicleTransformer(
                input_dim=len(data.columns) - 1,
                d_model=current_params.get('d_model', 128),
                nhead=current_params.get('nhead', 8),
                num_layers=current_params.get('num_layers', 6),
                dim_feedforward=current_params.get('dim_feedforward', 256)
            ).to(device)
            
            # Train model
            train_metrics, val_metrics, best_threshold, threshold_results = train_model(
                model,
                train_loader,
                test_loader,
                num_epochs=current_params.get('num_epochs', 2),
                device=device,
                lr=current_params.get('lr', 0.001)
            )
            
            # Get final predictions and metrics
            final_results = analyze_model_predictions(
                model, 
                test_loader, 
                scaler, 
                device, 
                best_threshold=best_threshold
            )
            
            # Store results
            result = {
                **current_params,
                'mileage_rmse' : final_results["metrics"]['mileage_rmse'],
                'mileage_mae' : final_results["metrics"]['mileage_mae'],
                'mileage_r2' : final_results["metrics"]['rsquared'],
                'scrap_accuracy' : final_results["metrics"]['scrap_accuracy'],
                'scap_auc' : final_results["metrics"]['scrap_auc'],
                'n_scrapped' : final_results["metrics"]['n_scrapped'],
                'n_pred_scrapped' : final_results["metrics"]['n_pred_scrapped'],
            }
            
            results.append(result)
            
            # Save intermediate results to CSV
            pd.DataFrame(results).to_csv(f"hyperparameter_tuning_results.csv", index=False)
            
        except Exception as e:
            print(f"Error with combination {i}: {str(e)}")
            continue
            
    return pd.DataFrame(results)

def validate_dataset(dataloader):
    for batch in dataloader:
        sequences, masks, batch_mileage, batch_scrapped = batch
        print(f"Sequence shape: {sequences.shape}")
        print(f"Mask shape: {masks.shape}")
        print(f"Mileage shape: {batch_mileage.shape}")
        print(f"Scrapped shape: {batch_scrapped.shape}")
        print(f"Scrapped range: [{batch_scrapped.min()}, {batch_scrapped.max()}]")
        break

def get_test_predictions(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader, 
    scaler: object,
    device: torch.device
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[str]]:
    """
    Get predictions for all test data and return both normalized and denormalized values,
    along with predictor variables.
    
    Column order in sequences:
    0-1: categorical ['fuel_type', 'last_test']
    2-5: numerical ['mileage_per_year', 'test_mileage', 'age_year', 'time_between_tests']
    """
    model.eval()
    all_mileage_preds = []
    all_scrapped_preds = []
    all_true_mileage = [] 
    all_true_scrapped = []
    all_ages = []
    all_fuel_types = []
    
    # Correct indices based on column order
    fuel_idx = 0  # First categorical column
    age_idx = 4   # Index for age_year (2 categorical + 2 numerical before age)
    
    with torch.no_grad():
        for sequences, masks, true_mileage, true_scrapped in tqdm(test_loader, desc="Predicting Test Data"):
            sequences = sequences.to(device)
            masks = masks.to(device)
            
            # Get valid sequence lengths from masks
            seq_lengths = masks.sum(dim=1).long() - 1  # -1 to get 0-based index
            batch_indices = torch.arange(sequences.size(0), device=device)
            
            # Get predictions
            mileage_pred, scrapped_pred = model(sequences, masks)
            
            # Get the age and fuel type from the last valid timestep of each sequence
            batch_ages = sequences[batch_indices, seq_lengths, age_idx].cpu().numpy()
            batch_fuel_types = sequences[batch_indices, seq_lengths, fuel_idx].cpu().numpy()
            
            all_mileage_preds.extend(mileage_pred.cpu().numpy())
            all_scrapped_preds.extend(scrapped_pred.cpu().numpy())
            all_true_mileage.extend(true_mileage.cpu().numpy())
            all_true_scrapped.extend(true_scrapped.cpu().numpy())
            all_ages.extend(batch_ages)
            all_fuel_types.extend(batch_fuel_types)
    
    # Get scaler parameters for mileage (first numerical feature)
    mileage_mean = scaler.mean_[0]
    mileage_scale = scaler.scale_[0]
    
    # Get scaler parameters for age_year (third numerical feature)
    age_mean = scaler.mean_[2]
    age_scale = scaler.scale_[2]
    
    # Denormalize predictions and values
    denormalized_preds = [(pred[0] * mileage_scale + mileage_mean) for pred in all_mileage_preds]
    denormalized_true = [(true[0] * mileage_scale + mileage_mean) for true in all_true_mileage]
    denormalized_ages = [(age * age_scale + age_mean) for age in all_ages]
    
    # Print age distribution for verification
    ages_rounded = [round(age) for age in denormalized_ages]
    age_counts = pd.Series(ages_rounded).value_counts().sort_index()
    print("\nAge distribution in predictions:")
    print(age_counts)
    
    # Convert scrapped predictions to list
    scrapped_preds = [pred[0] for pred in all_scrapped_preds]
    true_scrapped = [true[0] for true in all_true_scrapped]
    
    return denormalized_preds, scrapped_preds, denormalized_true, true_scrapped, denormalized_ages, all_fuel_types

def analyze_model_predictions(model, test_loader, scaler, device, best_threshold=0.5):
    """
    Complete analysis of model predictions including denormalized values,
    detailed statistics, and predictor variables
    """
    print("Loading best model and getting predictions...")
    model.load_state_dict(torch.load('best_model.pth'))
    model = model.to(device)
    
    # Get predictions using the updated function
    print("\nGenerating predictions on test data...")
    denorm_preds, scrapped_preds, denorm_true, true_scrapped, ages, fuel_types = get_test_predictions(
        model, test_loader, scaler, device
    )
    
    # Basic prediction statistics
    print("\nBasic Statistics:")
    print("----------------")
    mileage_stats = {
        'mean_pred': np.mean(denorm_preds),
        'mean_true': np.mean(denorm_true),
        'median_pred': np.median(denorm_preds),
        'median_true': np.median(denorm_true),
        'std_pred': np.std(denorm_preds),
        'std_true': np.std(denorm_true),
    }
    
    print(f"Mileage Predictions (denormalized):")
    print(f"  Mean Predicted: {mileage_stats['mean_pred']:,.0f} miles")
    print(f"  Mean Actual: {mileage_stats['mean_true']:,.0f} miles")
    print(f"  Median Predicted: {mileage_stats['median_pred']:,.0f} miles")
    print(f"  Median Actual: {mileage_stats['median_true']:,.0f} miles")
    print(f"  Std Dev Predicted: {mileage_stats['std_pred']:,.0f} miles")
    print(f"  Std Dev Actual: {mileage_stats['std_true']:,.0f} miles")
    
    # Error metrics
    mileage_rmse = np.sqrt(mean_squared_error(denorm_true, denorm_preds))
    mileage_mae = np.mean(np.abs(np.array(denorm_true) - np.array(denorm_preds)))
    rsquared = r2_score(denorm_true, denorm_preds)
    mileage_adjusted_r2 = 1 - (1 - rsquared) * (len(denorm_true) - 1) / (len(denorm_true) - 4 - 1)
    mileage_median_absolute_error = median_absolute_error(denorm_true, denorm_preds)
    
    
    print("\nError Metrics:")
    print("-------------")
    print(f"Mileage RMSE: {mileage_rmse:,.0f} miles")
    print(f"Mileage MAE: {mileage_mae:,.0f} miles")
    print(f"R-Squared: {rsquared:.4f}")
    print(f"Adjusted R-Squared: {mileage_adjusted_r2:.4f}")
    print(f"Median Absolute Error: {mileage_median_absolute_error:,.0f} miles")
    
    
    # Scrappage predictions
    print("\nScrappage Predictions:")
    print("---------------------")
    scrap_accuracy = accuracy_score(
        [1 if x > best_threshold else 0 for x in true_scrapped],
        [1 if x > best_threshold else 0 for x in scrapped_preds]
    )
    scrap_auc = roc_auc_score(true_scrapped, scrapped_preds)
    precision = precision_score(
        [1 if x > best_threshold else 0 for x in true_scrapped],
        [1 if x > best_threshold else 0 for x in scrapped_preds]
    )
    recall = recall_score(
        [1 if x > best_threshold else 0 for x in true_scrapped],
        [1 if x > best_threshold else 0 for x in scrapped_preds]
    )
    f1 = f1_score(
        [1 if x > best_threshold else 0 for x in true_scrapped],
        [1 if x > best_threshold else 0 for x in scrapped_preds]
    )
    
    
    print(f"Accuracy: {scrap_accuracy:.4f}")
    print(f"AUC-ROC: {scrap_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print(f"Average Predicted Scrap Probability: {np.mean(scrapped_preds):.2%}")
    print(f"Actual Scrap Rate: {np.mean(true_scrapped):.2%}")
    
    # Compare number of scrapped vehicles
    actual_scrapped = np.sum(true_scrapped)
    predicted_scrapped = np.sum([1 if x > best_threshold else 0 for x in scrapped_preds])
    print(f"\nActual Scrapped Vehicles: {actual_scrapped:,}")
    print(f"Predicted Scrapped Vehicles: {predicted_scrapped:,}")
    
    # Next print the age distribution of the scrapped vehicles
    scrapped_ages = [age for age, scrapped in zip(ages, true_scrapped) if scrapped == 1]
    scrapped_ages_rounded = [round(age) for age in scrapped_ages]
    scrapped_age_counts = pd.Series(scrapped_ages_rounded).value_counts().sort_index()
    print("\nAge distribution of scrapped vehicles:")
    print(scrapped_age_counts)
    
    # Followed by the predicted scrapped vehicles age distribution
    pred_scrapped_ages = [age for age, scrapped in zip(ages, scrapped_preds) if scrapped > best_threshold]
    pred_scrapped_ages_rounded = [round(age) for age in pred_scrapped_ages]
    pred_scrapped_age_counts = pd.Series(pred_scrapped_ages_rounded).value_counts().sort_index()
    print("\nAge distribution of predicted scrapped vehicles:")
    print(pred_scrapped_age_counts)
    
    return {
        'mileage_preds': denorm_preds,
        'mileage_true': denorm_true,
        'scrap_preds': scrapped_preds,
        'scrap_true': true_scrapped,
        'age_year': ages,
        'fuel_type': fuel_types,
        'metrics': {
            'mileage_rmse': mileage_rmse,
            'mileage_mae': mileage_mae,
            'scrap_accuracy': scrap_accuracy,
            'scrap_auc': scrap_auc,
            'rsquared': rsquared,
            'n_scrapped': actual_scrapped,
            'n_pred_scrapped': predicted_scrapped,
        }
    }
    
def plot_heatmaps_by_age(results, axes, age_bins, cmap):
    """
    Create hexbin plots for each age group using dictionary results and provided axes.
    """
    # Convert lists to numpy arrays for easier manipulation
    age_years = np.array(results['age_year'])
    true_vals = np.array(results['mileage_true']) * 1.60934  # Convert to km
    pred_vals = np.array(results['mileage_preds']) * 1.60934  # Convert to km
    
    
    age_bins.append(age_bins[-1] + 1)  # Add one more bin for the last group
    
    age_labels = [str(i) for i in age_bins]
    
    # Create age group assignments
    age_groups = np.zeros_like(age_years, dtype=object)
    for i, (lower, upper) in enumerate(zip(age_bins[:-1], age_bins[1:])):
        mask = (age_years >= lower) & (age_years < upper)
        age_groups[mask] = age_labels[i]
    
    for j, age_group in enumerate(age_labels[:-1]):
        ax = axes[j]
        
        # Filter data for this age group
        mask = (age_groups == age_group)
        group_true = true_vals[mask]
        group_pred = pred_vals[mask]
        
        if len(group_true) > 0:  # Only plot if we have data
            # Create hexbin plot
            hb = ax.hexbin(group_true, group_pred, 
                           gridsize=30, cmap=cmap,
                           extent=(0, 30_000, 0, 30_000))
            
            age_text = f"Age: {age_group}"
            ax.text(0.5, 0.95, age_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='center',
                    fontsize=7)
        
        # Set limits and labels regardless of data presence
        ax.set_xlim(0, 30_000)
        ax.set_ylim(0, 30_000)
        
        ax.set_xticks([0, 30_000])
        ax.set_yticks([0, 30_000])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        if j in [0, 4, 8, 12]:
            ax.set_yticklabels(["0", "30k"])
        if j in [12, 13, 14, 15]:
            ax.set_xticklabels(["0", "30k"])
            
        # Add single xlabel and ylabel
        if j == 13:
            # Offest to the right between axis 13 and 14:
            ax.text(1, -0.5, 'True Mileage (km)', ha='center', va='center', transform=ax.transAxes)
        if j == 8:
            # Offset upwards between ax 8 and 4
            ax.text(-0.5, 1, 'Predicted Mileage (km)', ha='center', va='center', rotation='vertical', transform=ax.transAxes)
            

    return axes

def plot_heatmap(results, ax, cmap):
    """
    Create hexbin plot 
    """
    # Convert lists to numpy arrays for easier manipulation
    true_vals = np.array(results['mileage_true'])*1.60934
    pred_vals = np.array(results['mileage_preds'])*1.60934
    
    # Create hexbin plot
    hb = ax.hexbin(true_vals, pred_vals, 
                    gridsize=30, cmap=cmap,
                    extent=(0, 30_000, 0, 30_000))
    
    # Calculate metrics for this group
    rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
    r2 = r2_score(true_vals, pred_vals)
            
    # Set limits and labels regardless of data presence
    ax.set_xlim(0, 30_000)
    ax.set_ylim(0, 30_000)
    ax.set_xticklabels(["0", "5k", "10k", "15k", "20k", "25k", "30k"])
    ax.set_yticklabels(["0", "5k", "10k", "15k", "20k", "25k", "30k"])

    # Add labels
    ax.set_xlabel('True Mileage (km)')
    ax.set_ylabel('Predicted Mileage (km)')
    
    return ax

def plot_histograms(results, axes, color_pred, color_true):
    """
    Create hexbin plot 
    """
    # Convert lists to numpy arrays for easier manipulation
    true_vals = np.array(results['mileage_true'])*1.60934
    pred_vals = np.array(results['mileage_preds'])*1.60934
    
    # True vales are first ax, predicted values are second

    # Create histogram plot between 0 and 30,000 km
    ax = axes[0]
    ax.hist(true_vals, bins=30, range=(0, 30_000), color=color_pred, alpha=0.5, edgecolor='white')
    
    ax.set_xlim(0, 30_000)
    ax.set_xticks([0, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    
    ax = axes[1]
    ax.hist(pred_vals, bins=30, range=(0, 30_000), color=color_true, alpha=0.5, orientation='horizontal', edgecolor='white')

    ax.set_ylim(0, 30_000)
    ax.set_yticks([0, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    ax.set_yticklabels([])
    ax.set_xticks([])
    # only keep the y axis on - remove the other spines
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return axes

# Now construct a function to plot the mean predicted mileage by age and fuel type for the real and predicted values
def plot_mileage(results, axes, color_pred, color_true):
    
    # Convert lists to numpy arrays for easier manipulation
    age_years = np.array(results['age_year'])
    fuel_types = np.array(results['fuel_type'])
    true_vals = np.array(results['mileage_true'])*1.60934
    pred_vals = np.array(results['mileage_preds'])*1.60934
    
    print(f"Mean true mileage: {np.mean(true_vals):,.0f} km")
    print(f"Mean predicted mileage: {np.mean(pred_vals):,.0f} km")
    print(f"Unique fuel types: {np.unique(fuel_types)}")
    
    # Create age groups
    age_bins = list(range(4, 31)) + [float('inf')]
    age_labels = [str(i) for i in range(4, 31)] + ['31+']
    
    # Create age group assignments
    age_groups = np.zeros_like(age_years, dtype=object)
    for i, (lower, upper) in enumerate(zip(age_bins[:-1], age_bins[1:])):
        mask = (age_years >= lower) & (age_years < upper)
        age_groups[mask] = age_labels[i]
    
    # Get unique fuel types
    unique_fuels = np.unique(fuel_types)
    
    for fuel_type, ax in zip(unique_fuels, axes):
        x_true = []
        y_true = []
        x_pred = []
        y_pred = []
        
        y_true_std = []
        y_pred_std = []
        
        y_true_75th = []
        y_pred_75th = []
        y_true_25th = []
        y_pred_25th = []
        
        y_true_50th = []
        y_pred_50th = []
        
        for age_group in age_labels:
            
            if age_group == '26':
                break

            # Filter data for this age group and fuel type
            mask = (age_groups == age_group) & (fuel_types == fuel_type)
            
            # check that we have data for this group
            if len(true_vals[mask]) == 0:
                continue
            
            group_true = true_vals[mask]
            group_pred = pred_vals[mask]
            if age_group != '31+':
                x_true.append(float(age_group))
                y_true.append(np.mean(group_true))
                x_pred.append(float(age_group))
                y_pred.append(np.mean(group_pred))
                
                y_true_std.append(np.std(group_true))
                y_pred_std.append(np.std(group_pred))
                
                y_true_75th.append(np.percentile(group_true, 75))
                y_pred_75th.append(np.percentile(group_pred, 75))
                y_true_25th.append(np.percentile(group_true, 25))
                y_pred_25th.append(np.percentile(group_pred, 25))
                
                y_true_50th.append(np.percentile(group_true, 50))
                y_pred_50th.append(np.percentile(group_pred, 50))
                
            else:
                x_true.append(31)
                y_true.append(np.mean(group_true))
                x_pred.append(31)
                y_pred.append(np.mean(group_pred))
                
                y_true_std.append(np.std(group_true))
                y_pred_std.append(np.std(group_pred))
                
                y_true_75th.append(np.percentile(group_true, 75))
                y_pred_75th.append(np.percentile(group_pred, 75))
                y_true_25th.append(np.percentile(group_true, 25))
                y_pred_25th.append(np.percentile(group_pred, 25))
                
                y_true_50th.append(np.percentile(group_true, 50))
                y_pred_50th.append(np.percentile(group_pred, 50))
                
        # Convert the std into two arrays for use in fill_between showing the sigma range
        y_true_upper = np.array(y_true) + np.array(y_true_std)
        y_true_lower = np.array(y_true) - np.array(y_true_std)
        
        y_pred_upper = np.array(y_pred) + np.array(y_pred_std)
        y_pred_lower = np.array(y_pred) - np.array(y_pred_std)
            
        # Plot the results
        ax.plot([0], [0], color='white', label="True Mileage")  # Add empty plot for label
        ax.plot(x_true, y_true, label='Mean', color=color_true)
        ax.plot(x_true, y_true_50th, label='Median', color=color_true, linestyle='--')
        ax.fill_between(x_true, y_true_75th, y_true_25th, color=color_true, alpha=0.15, label='25th-75th Percentile')
        
        ax.plot([0], [0], color='white', label="Predicted Mileage")  # Add empty plot for label
        ax.plot(x_pred, y_pred, label='Mean', color=color_pred)
        ax.plot(x_pred, y_pred_50th, label='Median', color=color_pred, linestyle='--')
        ax.fill_between(x_pred, y_pred_75th, y_pred_25th, color=color_pred, alpha=0.15, label='25th-75th Percentile')
        
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Annual Mileage (km)')
        
        ax.set_xlim(0,25)
        ax.set_ylim(0,30_000)
        
        ax.set_yticklabels(["0", "5k", "10k", "15k", "20k", "25k", "30k"])
        
        ax.legend()
        
    return axes
                  
def transformer_figure(results):
    
    # create custom cmap:
    cmap = mpl.colors.LinearSegmentedColormap.from_list("mycmap", ["white", "steelblue"])
    
    fig = plt.figure(figsize=(14,14), dpi=1000)

    # Add gridspec
    gs = fig.add_gridspec(9, 9)
    ax1 = fig.add_subplot(gs[0:4, 0:4])
    ax2 = fig.add_subplot(gs[0:4, 5:9])
    
    ax3 = fig.add_subplot(gs[6:9, 0:3])
    ax3_top = fig.add_subplot(gs[5:6, 0:3]) # ax3_top
    ax3_right = fig.add_subplot(gs[6:9, 3:4]) # ax3_right
    
    # In the space of gs[5:9, 5:9] add 4x4 plots
    ax4_00 = fig.add_subplot(gs[5:6, 5:6])
    ax4_01 = fig.add_subplot(gs[5:6, 6:7])
    ax4_02 = fig.add_subplot(gs[5:6, 7:8])
    ax4_03 = fig.add_subplot(gs[5:6, 8:9])
    ax4_10 = fig.add_subplot(gs[6:7, 5:6])
    ax4_11 = fig.add_subplot(gs[6:7, 6:7])
    ax4_12 = fig.add_subplot(gs[6:7, 7:8])
    ax4_13 = fig.add_subplot(gs[6:7, 8:9])
    ax4_20 = fig.add_subplot(gs[7:8, 5:6])
    ax4_21 = fig.add_subplot(gs[7:8, 6:7])
    ax4_22 = fig.add_subplot(gs[7:8, 7:8])
    ax4_23 = fig.add_subplot(gs[7:8, 8:9])
    ax4_30 = fig.add_subplot(gs[8:9, 5:6])
    ax4_31 = fig.add_subplot(gs[8:9, 6:7])
    ax4_32 = fig.add_subplot(gs[8:9, 7:8])
    ax4_33 = fig.add_subplot(gs[8:9, 8:9])
    
    # Plots A and B
    axes = [ax1, ax2]
    axes = plot_mileage(results, axes,  'steelblue', 'grey')
       
    # Plot C
    ax3 = plot_heatmap(results, ax3, cmap)
    
    [ax3_top, ax3_right] = plot_histograms(results, [ax3_top, ax3_right], 'grey', 'steelblue')
    
    # Plot D
    age_axes = [ax4_00, ax4_01, ax4_02, ax4_03, ax4_10, ax4_11, ax4_12, ax4_13, ax4_20, ax4_21, ax4_22, ax4_23, ax4_30, ax4_31, ax4_32, ax4_33]
    age_bins = list(range(4, 20))
    age_axes = plot_heatmaps_by_age(results, age_axes, age_bins, cmap)
    
    # Finally Add A, B, C, and D to the figure
    for c, ax in zip(['a', 'b', 'c',], [ax1, ax2, ax3_top]):
        ax.text(-0.15, 1.15, c, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='right')
        
    for c, ax in zip(['d'], [ ax4_00]):
        ax.text(-0.6, 1.25, c, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='right')
    
    return fig
  
class PredictionDataset(Dataset):
    """Dataset class for prediction sequences"""
    def __init__(self, sequences, vehicle_ids):
        self.sequences = sequences
        self.vehicle_ids = vehicle_ids
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.vehicle_ids[idx]

def collate_pred_batch(batch):
    """Custom collate function for prediction batches"""
    sequences, vehicle_ids = zip(*batch)
    padded_seqs, masks = pad_sequences(sequences)
    return torch.FloatTensor(padded_seqs), torch.FloatTensor(masks), list(vehicle_ids)

def prepare_prediction_data(df, label_encoders, scaler, max_seq_length=8):
    """Prepare data for prediction by creating sequences"""
    print("Preparing prediction data...")
    
    # Pre-define column lists
    categorical_cols = ['fuel_type', 'last_test' ]
    numerical_cols = ['mileage_per_year', 'test_mileage', 'age_year', 'time_between_tests']
    training_cols = ['vehicle_id'] + categorical_cols + numerical_cols
    
    # Process features
    df_processed = df[training_cols].copy()
    
    # Finally check for any NaN values and drop vehicle_ids that have them
    print("Dropping vehicles with NaN values...")
    n = df_processed["vehicle_id"].nunique()
    ids_nan = df_processed[df_processed.isnull().any(axis=1)]['vehicle_id'].unique()
    df_processed = df_processed[~df_processed['vehicle_id'].isin(ids_nan)]
    new_n = df_processed["vehicle_id"].nunique()
    print(f"Dropped {n - new_n} vehicles with NaN values.")
    
    # Get last tests before preprocessing for later use
    last_tests = df.sort_values('test_year').groupby('vehicle_id').last().reset_index()
    
    # Encode categorical variables
    for col in categorical_cols:
        df_processed[col] = label_encoders[col].transform(df_processed[col])
    
    # Scale numerical features
    df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])
    
    # Create sequences for each vehicle
    sequences = []
    vehicle_ids = []
    
    for vehicle_id, group in tqdm(df_processed.groupby('vehicle_id'), desc="Creating sequences"):
        group_data = group.drop('vehicle_id', axis=1).values
        
        # Use the last max_seq_length records for prediction
        if len(group_data) > max_seq_length:
            sequence = group_data[-max_seq_length:]
        else:
            sequence = group_data
            
        sequences.append(sequence)
        vehicle_ids.append(vehicle_id)
    
    return sequences, vehicle_ids, last_tests

def generate_predictions(model, data, label_encoders, scaler, device, batch_size=25_000, threshold=0.5):
    """Generate predictions for the next test for each vehicle"""
    print("Starting prediction generation...")
    model.eval()
    
    # Prepare data
    sequences, vehicle_ids, last_tests = prepare_prediction_data(data, label_encoders, scaler)
    
    # Create dataset and dataloader
    dataset = PredictionDataset(sequences, vehicle_ids)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pred_batch,
        num_workers=4,
        pin_memory=True
    )
    
    # Generate predictions
    print("Generating predictions...")
    all_predictions = []
    
    with torch.no_grad():
        for batch_sequences, batch_masks, batch_vehicle_ids in tqdm(dataloader, desc="Processing batches"):
            # Move to device
            batch_sequences = batch_sequences.to(device)
            batch_masks = batch_masks.to(device)
            
            # Get predictions
            mileage_preds, scrapped_preds = model(batch_sequences, batch_masks)
            
            # Move to CPU and convert to numpy
            mileage_preds = mileage_preds.cpu().numpy()
            scrapped_preds = scrapped_preds.cpu().numpy()
            
            # Denormalize mileage predictions
            mileage_preds = mileage_preds * scaler.scale_[0] + scaler.mean_[0]
            
            # Create predictions for this batch
            for idx, vehicle_id in enumerate(batch_vehicle_ids):
                last_test = last_tests[last_tests['vehicle_id'] == vehicle_id].iloc[0]
                
                new_row = last_test.copy()
                new_row['scrap_probability'] = scrapped_preds[idx][0]
                new_row['last_test'] = bool(scrapped_preds[idx][0] > threshold)
                new_row['mileage_per_year'] = mileage_preds[idx][0]
                new_row['test_mileage'] = last_test['test_mileage'] + mileage_preds[idx][0]
                new_row['age_year'] = last_test['age_year'] + 1
                new_row['time_between_tests'] = 1.0
                new_row['test_year'] = new_row['test_year'] + 1
                new_row['simulated_data'] = True
                
                all_predictions.append(new_row)
                
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
    # Create predictions DataFrame
    print("Creating final DataFrame...")
    pred_df = pd.DataFrame(all_predictions)
    
    return pred_df            

def plot_mileage_predictions(data, title):
    
    # Figure for HEVs and BEVs
    fuels = ["HY", "EL", "DI", "PE"]
    fuel_colours = ["#ef476f", "#ffd166","#118ab2", "#073b4c"]
    fuel_colors_lookup = {fuel : color for fuel, color in zip(fuels, fuel_colours)}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    ax = ax1
    for fuel in fuels:
        
        fuel_data = data[data['fuel_type'] == fuel]
        
        x = []
        y = [] 
        
        for age in range(1, 26, 1):
            
            if len(fuel_data[fuel_data['age_year'] == age]) != 0:
                x.append(age)
                y.append(fuel_data[fuel_data['age_year'] == age]['mileage_per_year'].mean())
            else:
                continue
            
        ax.plot(x, y, label=fuel, color=fuel_colors_lookup[fuel], linewidth=2)
        
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 30_000)
        
        ax.legend()
        
    ax = ax2
    for fuel in fuels:
        
        fuel_data = data[data['fuel_type'] == fuel]
        
        x = []
        y = [] 
        
        for age in range(1, 26, 1):
            
            if len(fuel_data[fuel_data['age_year'] == age]) != 0:
                x.append(age)
                y.append(fuel_data[fuel_data['age_year'] == age]['test_mileage'].mean())
            else:
                continue
            
        ax.plot(x, y, label=fuel, color=fuel_colors_lookup[fuel], linewidth=2)
        
        x = []
        y = [] 
        
        for age in range(1, 26, 1):
            
            if len(fuel_data[fuel_data['age_year'] == age & (fuel_data["last_test"]==True)]) != 0:
                x.append(age)
                y.append(fuel_data[(fuel_data['age_year'] == age) & (fuel_data["last_test"]==True)]['test_mileage'].mean())
            else:
                continue
            
        ax.plot(x, y, label=f"{fuel} Scrappage Mileage", color=fuel_colors_lookup[fuel], linewidth=2, linestyle='--')   
        
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 300_000)
    
        ax.legend()
    
    plt.suptitle(title)
    
    plt.show()
    
    return fig   

