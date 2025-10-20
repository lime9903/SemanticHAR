"""
MARBLE and UCI ADL dataset loader
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import random
from tqdm import tqdm
import json
from datetime import datetime

sys.path.append('/workspace/semantic')
from config import SemanticHARConfig

def parse_datetime(series):
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S ', '%Y-%m-%d %H:%M:%S\t']:
        try:
            return pd.to_datetime(series, format=fmt, errors='coerce')
        except:
            continue

    return pd.to_datetime(series, errors='coerce')

class SensorDataset:
    """Sensor Dataset Class"""
    
    def __init__(self, config: SemanticHARConfig):
        """
        Args:
            config: Configuration
            source_dataset: Source dataset name ("MARBLE" or "UCI_ADL_home_a" or "UCI_ADL_home_b")
            target_dataset: Target dataset name ("MARBLE" or "UCI_ADL_home_a" or "UCI_ADL_home_b")
        """
        self.config = config
        self.source_dataset = config.source_dataset
        self.target_dataset = config.target_dataset
        self.source_sensor_data, self.source_activity_label, self.source_activity_name, self.target_sensor_data, self.target_activity_label, self.target_activity_name = self._load_data()

    
    def _load_data(self) -> Tuple[Tuple[List[pd.DataFrame], List[str], List[str]], Tuple[List[pd.DataFrame], List[str], List[str]]]:
        """Data loading"""

        # Source data loading
        if self.source_dataset == "MARBLE":
            source_sensor_data, source_activity_label, source_activity_name = self._load_marble_data()
        elif self.source_dataset.startswith("UCI_ADL"):
            source_sensor_data, source_activity_label, source_activity_name = self._load_uci_adl_data(self.source_dataset)
        else:
            raise ValueError(f"✗ Unsupported dataset: {self.source_dataset}")

        # Target data loading
        if self.target_dataset == "MARBLE":
            target_sensor_data, target_activity_label, target_activity_name = self._load_marble_data()
        elif self.target_dataset.startswith("UCI_ADL"):
            target_sensor_data, target_activity_label, target_activity_name = self._load_uci_adl_data(self.target_dataset)
        else:
            raise ValueError(f"✗ Unsupported dataset: {self.target_dataset}")
        
        return source_sensor_data, source_activity_label, source_activity_name, target_sensor_data, target_activity_label, target_activity_name
    
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
    
    def _load_uci_adl_data(self, dataset_name: str = "UCI_ADL_home_a") -> Tuple[List[pd.DataFrame], List[str], List[str]]:
        """UCI ADL Dataset loading"""
        activity_label = None
        activity_name = None
        data_path = self.config.uci_adl_data_path

        # Determine which dataset to load based on dataset_name
        if dataset_name == "UCI_ADL_home_a":
            adl_file = 'OrdonezA_ADLs.txt'
        elif dataset_name == "UCI_ADL_home_b":
            adl_file = 'OrdonezB_ADLs.txt'
        else:
            raise ValueError(f"✗ Unsupported dataset: {dataset_name}")

        # Load ADL data
        try: 
            adl_path = os.path.join(data_path, adl_file)
            if not os.path.exists(adl_path):
                raise ValueError(f"✗ ADL file not found: {adl_path}")
                    
            adl_df = self._load_adl_file(adl_path)
            if adl_df is None or len(adl_df) == 0:
                raise ValueError(f"✗ ADL data is empty: {adl_path}")
            
            # Load corresponding sensor data
            sensor_file = adl_file.replace('_ADLs.txt', '_Sensors.txt')
            sensor_path = os.path.join(data_path, sensor_file)
            if not os.path.exists(sensor_path):
                raise ValueError(f"✗ Sensor file not found: {sensor_path}")
            
            sensor_df = self._load_sensor_file(sensor_path)
            if sensor_df is None or len(sensor_df) == 0:
                raise ValueError(f"✗ Sensor data is empty: {sensor_path}")
            
            sensor_data_df = pd.DataFrame()
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
                    data = {
                        'timestamp': sensor_time,
                        'sensor_location': sensor_row['location'],
                        'sensor_type': sensor_row['type'],
                        'sensor_place': sensor_row['place'],
                        'activity': matching_activity,
                        'sensor_duration': (pd.to_datetime(sensor_row['end_time']) - sensor_time).total_seconds()
                    }
                    sensor_data_df = pd.concat([sensor_data_df, pd.DataFrame([data])], ignore_index=True)
            
            if len(sensor_data_df) > 0:
                activity_label = np.array(sensor_data_df['activity'])
                activity_name = sensor_data_df['activity'].unique().tolist()
                print(f"\nCreated UCI ADL dataset with {len(sensor_data_df)} sensor-activity pairs from {dataset_name}")
                print(f"Sensor data: {sensor_data_df[:3]}")
                print(f"Activity names: {activity_name}")

        except Exception as e:
            print(f"✗ UCI ADL file loading error {adl_file}: {e}")
            return None, None, None
    
        return sensor_data_df, activity_label, activity_name
    
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
    
    def get_data(self):
        return self.source_sensor_data, self.source_activity_label, self.source_activity_name, self.target_sensor_data, self.target_activity_label, self.target_activity_name

def load_sensor_data(config: SemanticHARConfig, use_event_based: bool = True) -> Dict[str, Dict[str, List[pd.DataFrame]]]:
    """Load sensor data as DataFrames and split with event-based or time-based windows"""

    dataset_loader = SensorDataset(config)
    source_sensor_data, _, _, target_sensor_data, _, _ = dataset_loader.get_data()

    split_data = {
        'train': {},
        'val': {},
        'test': {}
    }
    
    # Process source dataset (train + val)
    print(f"\nProcessing source dataset: {config.source_dataset}")
    
    if use_event_based:
        print(f"  Processing {len(source_sensor_data)} events with event-based grouping...")
        # Create event-based windows for source data
        source_windows = _create_event_based_windows(source_sensor_data)
        print(f"  Created {len(source_windows)} source event-based windows")
    else:
        print(f"  Processing {len(source_sensor_data)} events with {config.window_size_seconds}s windows, {config.overlap_ratio*100}% overlap...")
        # Create time windows for source data
        source_windows = _create_time_based_windows(source_sensor_data, config.window_size_seconds, config.overlap_ratio)
        print(f"  Created {len(source_windows)} source time windows")
    
    # Split source data: 80% train, 20% val
    train_windows, val_windows = _split_windows(source_windows, config.source_train_ratio)
    split_data['train'][config.source_dataset] = train_windows
    split_data['val'][config.source_dataset] = val_windows
    
    print(f"  → train:   {len(train_windows)} windows ({len(train_windows)/len(source_windows)*100:.2f}%)")
    print(f"  → val:     {len(val_windows)} windows ({len(val_windows)/len(source_windows)*100:.2f}%)")
    
    # Process target dataset (test)
    print(f"\nProcessing target dataset: {config.target_dataset}")
    
    if use_event_based:
        print(f"  Processing {len(target_sensor_data)} events with event-based grouping...")
        # Create event-based windows for target data
        target_windows = _create_event_based_windows(target_sensor_data)
        print(f"  Created {len(target_windows)} target event-based windows")
    else:
        print(f"  Processing {len(target_sensor_data)} events with {config.window_size_seconds}s windows, {config.overlap_ratio*100}% overlap...")
        # Create time windows for target data
        target_windows = _create_time_based_windows(target_sensor_data, config.window_size_seconds, config.overlap_ratio)
        print(f"  Created {len(target_windows)} target time windows")
    
    # Target data: all windows for test
    split_data['test'][config.target_dataset] = target_windows
    
    print(f"  → test:    {len(target_windows)} windows ({len(target_windows)/len(target_windows)*100:.2f}%)")
    
    print("\n" + "-" * 64)
    print("DATA STRUCTURE SUMMARY")
    for split_name in ['train', 'val', 'test']:
        source_count = len(split_data[split_name].get(config.source_dataset, []))
        target_count = len(split_data[split_name].get(config.target_dataset, []))
        total = source_count + target_count
        print(f"  {split_name:15} : {total:5} windows (source: {source_count:4}, target: {target_count:4})")
    print("-" * 64 + "\n")

    # Save windows to JSON
    _save_windows(split_data, config, use_event_based)
    
    return split_data


def _create_event_based_windows(df: pd.DataFrame) -> List[pd.DataFrame]:
    """Create event-based windows"""
    
    # Sort by timestamp
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    if len(df_sorted) == 0:
        return []
    
    print(f"    Creating event-based windows")
    print(f"    Total events: {len(df_sorted)}")
    
    windows = []
    
    for idx, row in df_sorted.iterrows():
        window_df = pd.DataFrame([row]).copy()

        window_df['window_start'] = row['timestamp']
        window_df['window_end'] = row['timestamp'] + pd.Timedelta(seconds=row['sensor_duration'])
        window_df['window_duration'] = row['sensor_duration']
        
        windows.append(window_df)
    
    print(f"    Created {len(windows)} event-based windows")
    for i, window in enumerate(windows[:5]):  # Show first 5 windows as examples
        print(f"      Window {i+1}: {window['window_duration'].iloc[0]:.1f}s, "
              f"activity: {window['activity'].iloc[0]}")
    
    return windows


def _create_time_based_windows(df: pd.DataFrame, window_size_seconds: int, overlap_ratio: float) -> List[pd.DataFrame]:
    """Create time windows based on sensor events duration"""
    window_size_seconds = int(window_size_seconds)
    overlap_ratio = float(overlap_ratio)
    
    # Sort by timestamp
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    if len(df_sorted) == 0:
        return []
    
    # Get time range
    start_time = df_sorted['timestamp'].min()
    end_time = df_sorted['timestamp'].max()
    total_duration = (end_time - start_time).total_seconds()
    print(f"    Time range: {start_time} to {end_time} (duration: {total_duration:.1f}s)")
    
    step_size = window_size_seconds * (1 - overlap_ratio)
    print(f"    Window size: {window_size_seconds}s, Overlap: {overlap_ratio*100:.1f}%, Step size: {step_size:.1f}s")
    
    windows = []
    current_time = start_time
    
    # Calculate total number of windows for progress bar
    total_windows = int((end_time - start_time).total_seconds() / step_size) + 1
    pbar = tqdm(total=total_windows, desc="    Creating windows", leave=False)
    
    while current_time < end_time:
        window_end = current_time + pd.Timedelta(seconds=window_size_seconds)
        
        # Get events that overlap with this time window
        # An event overlaps if: event_start < window_end AND event_end > window_start
        window_events = df_sorted[
            (df_sorted['timestamp'] < window_end) & 
            (df_sorted['timestamp'] + pd.to_timedelta(df_sorted['sensor_duration'], unit='s') > current_time)
        ].copy()
        
        window_events['window_start'] = current_time
        window_events['window_end'] = window_end
        window_events['window_duration'] = window_size_seconds
        
        # Determine activity for this window
        if len(window_events) > 0:
            window_activity = window_events['activity'].mode().iloc[0] if len(window_events['activity'].mode()) > 0 else 'Unknown'
        else:
            past_events = df_sorted[df_sorted['timestamp'] <= current_time]
            if len(past_events) > 0:
                # Find the most recent event that might still be active
                for _, event in past_events.iloc[::-1].iterrows():
                    event_end = event['timestamp'] + pd.to_timedelta(event['sensor_duration'], unit='s')
                    if event_end > current_time:
                        window_activity = event['activity']
                        break
                else:
                    window_activity = 'Unknown'
            else:
                window_activity = 'Unknown'
        
        # Add activity to all rows in this window
        window_events['activity'] = window_activity
        
        windows.append(window_events)
        pbar.update(1)
        current_time += pd.Timedelta(seconds=step_size)
    
    pbar.close()
    print(f"    Created {len(windows)} windows")

    return windows


def _save_windows(split_data: Dict, config, use_event_based: bool = True) -> str:
    """Save windows to JSON file"""
    
    # Base generation info
    generation_info = {
        'timestamp': datetime.now().isoformat(),
        'source_dataset': config.source_dataset,
        'target_dataset': config.target_dataset,
    }
    
    # Add generation information for each generation strategy
    if not use_event_based:
        generation_info['window_size_seconds'] = config.window_size_seconds
        generation_info['overlap_ratio'] = config.overlap_ratio
        generation_info['generation_strategy'] = 'time-based'
    else:
        generation_info['generation_strategy'] = 'event-based'
    
    windows_data = {
        'generation_info': generation_info,
        'windows': {
            'train': [],
            'val': [],
            'test': []
        }
    }
    
    for split_name in ['train', 'val', 'test']:
        if split_name in split_data:
            for dataset in [config.source_dataset, config.target_dataset]:
                windows = split_data[split_name].get(dataset, [])
                for i, window in enumerate(windows):
                    window_start = str(window['window_start'].iloc[0]) if 'window_start' in window.columns else 'N/A'
                    window_end = str(window['window_end'].iloc[0]) if 'window_end' in window.columns else 'N/A'
                    window_duration = window['window_duration'].iloc[0] if 'window_duration' in window.columns else 'N/A'
                    activity = window['activity'].iloc[0] if 'activity' in window.columns else 'Unknown'
                    sensor_types = window['sensor_type'].unique().tolist() if 'sensor_type' in window.columns else []
                    locations = window['sensor_location'].unique().tolist() if 'sensor_location' in window.columns else []
                    places = window['sensor_place'].unique().tolist() if 'sensor_place' in window.columns else []
                    
                    window_info = {
                        'window_id': f'{dataset}_{split_name}_{i+1}',
                        'window_start': window_start,
                        'window_end': window_end,
                        'window_duration': window_duration,
                        'activity': activity,
                        'sensor_types': sensor_types,
                        'locations': locations,
                        'places': places
                    }
                    windows_data['windows'][split_name].append(window_info)
    
    with open(config.windows_file, 'w', encoding='utf-8') as f:
        json.dump(windows_data, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"✓ Windows saved to: {config.windows_file}")
    return config.windows_file


def _split_windows(windows: List[pd.DataFrame], train_ratio: float) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Split windows using percentages
    
    Returns:
        train: train_ratio
    """
    total_windows = len(windows)
    train_count = int(total_windows * train_ratio)
    
    # Shuffle windows
    random.seed(42)
    random.shuffle(windows)
    
    # Split into train and val
    train = windows[:train_count]
    val = windows[train_count:]
    
    return train, val
