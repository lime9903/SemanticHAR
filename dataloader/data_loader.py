"""
MARBLE and UCI ADL dataset loader
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime, timedelta
import re

sys.path.append('/workspace/semantic')
from config import SemanticHARConfig

def parse_datetime(series):
    # Handle datetime parsing with multiple format attempts
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S ', '%Y-%m-%d %H:%M:%S\t']:
        try:
            return pd.to_datetime(series, format=fmt, errors='coerce')
        except:
            continue
    # If all formats fail, use flexible parsing
    return pd.to_datetime(series, errors='coerce')

class SensorDataset:
    """Sensor Dataset Class - DataFrame을 직접 반환"""
    
    def __init__(self, data_path: str, dataset_name: str):
        """
        Args:
            data_path: Dataset path
            dataset_name: Dataset name ("MARBLE" or "UCI_ADL")
        """
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.home_ids = ['home_a', 'home_b'] if dataset_name == "UCI_ADL" else None

        self.sensor_data, self.activity_labels, self.activity_names = self._load_data()
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Data loading"""
        if self.dataset_name == "MARBLE":
            return self._load_marble_data()
        elif self.dataset_name == "UCI_ADL":
            return self._load_uci_adl_data()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def _load_marble_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """MARBLE Dataset loading"""
        data_files = []
        labels = []
        activities = []
        
        # MARBLE Dataset structure: dataset/{activity_code}/instance{num}/environmental.csv
        dataset_path = os.path.join(self.data_path, "dataset")
        
        if not os.path.exists(dataset_path):
            print(f"MARBLE Dataset path not found: {dataset_path}")
            return np.array([]), np.array([]), []
        
        # Activity code mapping (MARBLE code -> activity name)
        activity_mapping = {
            'A1a': 'sleeping', 'A1e': 'sleeping', 'A1m': 'sleeping',
            'A2a': 'toileting', 'A2e': 'toileting', 'A2m': 'toileting',
            'A4mae': 'toileting',
            'B1a': 'showering', 'B1e': 'showering', 'B1m': 'showering',
            'B2a': 'grooming', 'B2e': 'grooming', 'B2m': 'grooming',
            'B4mae': 'grooming',
            'C1a': 'breakfast', 'C1e': 'breakfast', 'C1m': 'breakfast',
            'C2a': 'lunch', 'C2e': 'lunch', 'C2m': 'lunch',
            'C4mae': 'dinner',
            'D1a': 'spare_time', 'D1e': 'spare_time', 'D1m': 'spare_time',
            'D4mae': 'spare_time'
        }
        
        for activity_code in os.listdir(dataset_path):
            if activity_code in activity_mapping:
                activity_name = activity_mapping[activity_code]
                activity_path = os.path.join(dataset_path, activity_code)
                
                if os.path.isdir(activity_path):
                    # Process each instance
                    for instance_dir in os.listdir(activity_path):
                        instance_path = os.path.join(activity_path, instance_dir)
                        if os.path.isdir(instance_path):
                            env_file = os.path.join(instance_path, "environmental.csv")
                            
                            if os.path.exists(env_file):
                                try:
                                    # Load environmental sensor data
                                    df = pd.read_csv(env_file)
                                    
                                    # Process sensor data
                                    sensor_data = self._process_marble_sensor_data(df)
                                    
                                    if len(sensor_data) > 0:
                                        data_files.append(sensor_data)
                                        labels.append(len(activities))
                                        activities.append(activity_name)
                                        
                                except Exception as e:
                                    print(f"File loading error {env_file}: {e}")
                                    continue
        
        # Window sliding to split data
        windowed_data = []
        windowed_labels = []
        
        for data, label in zip(data_files, labels):
            if len(data) >= self.window_size:
                step_size = int(self.window_size * (1 - self.overlap))
                for i in range(0, len(data) - self.window_size + 1, step_size):
                    window = data[i:i + self.window_size]
                    windowed_data.append(window)
                    windowed_labels.append(label)
        
        return np.array(windowed_data), np.array(windowed_labels), activities
    
    def _process_marble_sensor_data(self, df: pd.DataFrame) -> np.ndarray:
        """MARBLE Sensor data preprocessing"""
        # Sort timestamps
        df = df.sort_values('ts')
        
        # Separate data by sensor ID and extract features
        sensor_features = []
        
        # Unique sensor IDs
        unique_sensors = df['sensor_id'].unique()
        
        # Extract features for each sensor
        for sensor_id in unique_sensors:
            sensor_data = df[df['sensor_id'] == sensor_id].copy()
            
            # Track sensor status changes
            sensor_data['status_numeric'] = (sensor_data['sensor_status'] == 'ON').astype(int)
            
            # Time-based features
            sensor_data['time_diff'] = sensor_data['ts'].diff().fillna(0)
            sensor_data['duration'] = sensor_data['time_diff']
            
            # Sensor-based statistical features
            features = [
                sensor_data['status_numeric'].mean(),  # ON ratio
                sensor_data['status_numeric'].std(),   # Standard deviation of status changes
                sensor_data['duration'].mean(),        # Mean duration
                sensor_data['duration'].std(),         # Standard deviation of duration
                len(sensor_data),                      # Number of events
                sensor_data['status_numeric'].sum()    # Total ON time
            ]
            
            sensor_features.extend(features)
        
        # Pad to fixed dimension (6 features per sensor, maximum 20 sensors)
        max_sensors = 20
        features_per_sensor = 6
        max_features = max_sensors * features_per_sensor
        
        if len(sensor_features) < max_features:
            sensor_features.extend([0] * (max_features - len(sensor_features)))
        else:
            sensor_features = sensor_features[:max_features]
        
        return np.array(sensor_features)
    
    def _load_uci_adl_data(self) -> Tuple[List[pd.DataFrame], List[str], List[str]]:
        """UCI ADL Dataset loading"""
        sensor_files = {}
        activity_labels = {}
        activity_names = {}
        
        # Process UCI ADL Dataset files
        adl_files = [f for f in os.listdir(self.data_path) if f.endswith('_ADLs.txt')]
        
        for idx, adl_file in enumerate(adl_files):
            try:
                # Load ADL data
                adl_path = os.path.join(self.data_path, adl_file)
                if not os.path.exists(adl_path):
                    continue
                    
                adl_df = self._load_adl_file(adl_path)
                if adl_df is None or len(adl_df) == 0:
                    continue
                
                # Load corresponding sensor data
                sensor_file = adl_file.replace('_ADLs.txt', '_Sensors.txt')
                sensor_path = os.path.join(self.data_path, sensor_file)
                
                if not os.path.exists(sensor_path):
                    continue
                    
                sensor_df = self._load_sensor_file(sensor_path)
                if sensor_df is None or len(sensor_df) == 0:
                    continue
                
                # Create integrated dataframe by matching sensor events with activities
                sensor_data_df = []
        
                for _, sensor_row in sensor_df.iterrows():
                    sensor_time = pd.to_datetime(sensor_row['start_time'])
                    
                    # Find the activity that was happening at this sensor time
                    matching_activity = None
                    for _, activity_row in adl_df.iterrows():
                        activity_start = pd.to_datetime(activity_row['start_time'])
                        activity_end = pd.to_datetime(activity_row['end_time'])
                        
                        if activity_start <= sensor_time <= activity_end:
                            matching_activity = activity_row['activity'].strip()
                            break
                    
                    if matching_activity:
                        # Create integrated row
                        sensor_data = {
                            'timestamp': sensor_time,
                            'sensor_location': sensor_row['location'],
                            'sensor_type': sensor_row['type'],
                            'sensor_place': sensor_row['place'],
                            'activity': matching_activity,
                            'sensor_duration': (pd.to_datetime(sensor_row['end_time']) - sensor_time).total_seconds()
                        }
                        sensor_data_df.append(sensor_data)
        
                sensor_data_df = pd.DataFrame(sensor_data_df)
                
                if len(sensor_data_df) > 0:
                    sensor_files[self.home_ids[idx]] = sensor_data_df
                    activity_labels[self.home_ids[idx]] = sensor_data_df['activity']
                    activity_names[self.home_ids[idx]] = list(sensor_data_df['activity'].unique())
                    print(f"Created UCI ADL dataset with {len(sensor_data_df)} sensor-activity pairs from {self.home_ids[idx]}")
                    print(f"Sensor data: {sensor_data_df[:5]}")
                    print(f"Activity names: {activity_names[self.home_ids[idx]]}")

            except Exception as e:
                print(f"UCI ADL file loading error {adl_file}: {e}")
                continue
            
        return sensor_files, activity_labels, activity_names
    
    def _load_adl_file(self, adl_path: str) -> Optional[pd.DataFrame]:
        """Load ADL file"""
        try:
            with open(adl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Skip header lines
            data_lines = lines[2:]
            parsed_data = []
            
            for line in data_lines:
                line = line.strip()
                if not line:
                    continue

                if '\t' in line:
                    parts = [part.strip() for part in line.split('\t') if part.strip()]
                    if len(parts) >= 3:
                        start_time, end_time, activity = parts[0], parts[1], parts[2]
                        if start_time and end_time and activity:
                            parsed_data.append([start_time, end_time, activity])

            if not parsed_data:
                return None
            
            # Create DataFrame
            adl_df = pd.DataFrame(parsed_data, columns=['start_time', 'end_time', 'activity'])
            adl_df = adl_df.dropna()
            
            # Clean and parse datetime
            adl_df['start_time'] = adl_df['start_time'].astype(str).str.strip()
            adl_df['end_time'] = adl_df['end_time'].astype(str).str.strip()
            adl_df['activity'] = adl_df['activity'].astype(str).str.strip()
            
            # Remove empty rows
            adl_df = adl_df[(adl_df['start_time'] != '') & (adl_df['end_time'] != '') & (adl_df['activity'] != '')]
            
            adl_df['start_time'] = parse_datetime(adl_df['start_time'])
            adl_df['end_time'] = parse_datetime(adl_df['end_time'])

            return adl_df
            
        except Exception as e:
            print(f"Error loading ADL file {adl_path}: {e}")
            return None
    
    def _load_sensor_file(self, sensor_path: str) -> Optional[pd.DataFrame]:
        """Load sensor file"""
        try:
            with open(sensor_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Skip header lines
            data_lines = lines[2:]
            parsed_data = []
            
            for line in data_lines:
                line = line.strip()
                if not line:
                    continue
                
                if '\t' in line:
                    parts = [part.strip() for part in line.split('\t') if part.strip()]
                    if len(parts) >= 5:
                        start_time, end_time, location, sensor_type, place = parts[0], parts[1], parts[2], parts[3], parts[4]
                        if start_time and end_time and location and sensor_type and place:
                            parsed_data.append([start_time, end_time, location, sensor_type, place])

            if not parsed_data:
                return None
            
            # Create DataFrame
            sensor_df = pd.DataFrame(parsed_data, columns=['start_time', 'end_time', 'location', 'type', 'place'])
            sensor_df = sensor_df.dropna()
            
            # Clean and parse datetime
            sensor_df['start_time'] = sensor_df['start_time'].astype(str).str.strip()
            sensor_df['end_time'] = sensor_df['end_time'].astype(str).str.strip()
            sensor_df['location'] = sensor_df['location'].astype(str).str.strip()
            sensor_df['type'] = sensor_df['type'].astype(str).str.strip()
            sensor_df['place'] = sensor_df['place'].astype(str).str.strip()
            
            # Remove empty rows
            sensor_df = sensor_df[(sensor_df['start_time'] != '') & (sensor_df['end_time'] != '')]
            
            sensor_df['start_time'] = parse_datetime(sensor_df['start_time'])
            sensor_df['end_time'] = parse_datetime(sensor_df['end_time'])
            
            return sensor_df
            
        except Exception as e:
            print(f"Error loading sensor file {sensor_path}: {e}")
            return None
    
    def get_activity_names(self):
        return self.activity_names

def load_sensor_data(config: SemanticHARConfig, dataset_name: str, 
                    window_size_seconds: int = 60, overlap_ratio: float = 0.8) -> Dict[str, Dict[str, List[pd.DataFrame]]]:
    """Load sensor data as DataFrames and split into train/val/test with time windows for LLM processing"""
    
    data_path = config.marble_data_path if dataset_name == "MARBLE" else config.uci_adl_data_path
    dataset = SensorDataset(data_path, dataset_name)
    
    print(f"Loaded sensor data for: {list(dataset.sensor_data.keys())}")
    
    # Split each home's data into train/val/test with time windows
    split_data = {}
    for home_id, df in dataset.sensor_data.items():
        print(f"Processing {home_id} data ({len(df)} events) with {window_size_seconds}s windows, {overlap_ratio*100}% overlap...")
        
        # First split by time windows
        windows = _create_time_windows(df, window_size_seconds, overlap_ratio)
        print(f"  Created {len(windows)} time windows")
        
        # Then split windows into train/val/test (maintaining temporal order)
        train_windows, val_windows, test_windows = _split_windows(windows, train_ratio=0.7, val_ratio=0.15)
        
        split_data[home_id] = {
            'train': train_windows,
            'val': val_windows,
            'test': test_windows
        }
        
        print(f"  Train: {len(train_windows)} windows")
        print(f"  Val: {len(val_windows)} windows") 
        print(f"  Test: {len(test_windows)} windows")
    
    return split_data

def _split_dataframe(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train/val/test"""
    total_size = len(df)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Create indices
    indices = list(range(total_size))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create split DataFrames
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    
    return train_df, val_df, test_df

def _create_time_windows(df: pd.DataFrame, window_size_seconds: int, overlap_ratio: float) -> List[pd.DataFrame]:
    """Create time windows from sensor data"""
    # Sort by timestamp
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    if len(df_sorted) == 0:
        return []
    
    # Get time range
    start_time = df_sorted['timestamp'].min()
    end_time = df_sorted['timestamp'].max()
    
    # Calculate step size based on overlap
    step_size = window_size_seconds * (1 - overlap_ratio)
    
    windows = []
    current_time = start_time
    
    while current_time < end_time:
        window_end = current_time + pd.Timedelta(seconds=window_size_seconds)
        
        # Get events in this time window
        window_events = df_sorted[
            (df_sorted['timestamp'] >= current_time) & 
            (df_sorted['timestamp'] < window_end)
        ].copy()
        
        if len(window_events) > 0:
            # Add window metadata
            window_events['window_start'] = current_time
            window_events['window_end'] = window_end
            window_events['window_duration'] = window_size_seconds
            windows.append(window_events)
        
        # Move to next window
        current_time += pd.Timedelta(seconds=step_size)
    
    return windows

def _split_windows(windows: List[pd.DataFrame], train_ratio=0.7, val_ratio=0.15) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    """Split windows into train/val/test with shuffling"""
    total_windows = len(windows)
    train_size = int(total_windows * train_ratio)
    val_size = int(total_windows * val_ratio)
    
    # Shuffle windows - each window already contains temporal information
    # This allows the model to learn from diverse time periods
    import random
    random.seed(42)
    shuffled_windows = windows.copy()
    random.shuffle(shuffled_windows)
    
    # Split shuffled windows
    train_windows = shuffled_windows[:train_size]
    val_windows = shuffled_windows[train_size:train_size + val_size]
    test_windows = shuffled_windows[train_size + val_size:]
    
    return train_windows, val_windows, test_windows
