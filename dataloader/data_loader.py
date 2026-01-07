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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
            source_dataset: Source dataset name ("casas_aruba" / "casas_cairo" / "casas_kyoto" / "casas_milan" / "uci_adl_home_a" / "uci_adl_home_b")
            target_dataset: Target dataset name ("casas_aruba" / "casas_cairo" / "casas_kyoto" / "casas_milan" / "uci_adl_home_a" / "uci_adl_home_b")
        """
        self.config = config
        self.source_dataset = config.source_dataset
        self.target_dataset = config.target_dataset
        self.window_size = config.window_size_seconds
        self.overlap = config.overlap_ratio
        
        self.source_sensor_data, self.source_activity_label, self.source_activity_name, self.target_sensor_data, self.target_activity_label, self.target_activity_name = self._load_data()
    
    def _load_data(self) -> Tuple[Tuple[List[pd.DataFrame], List[str], List[str]], Tuple[List[pd.DataFrame], List[str], List[str]]]:
        """Data loading"""

        # Source data loading
        source_sensor_data, source_activity_label, source_activity_name = self._load_dataset(self.source_dataset)

        # Target data loading
        target_sensor_data, target_activity_label, target_activity_name = self._load_dataset(self.target_dataset)
        
        return source_sensor_data, source_activity_label, source_activity_name, target_sensor_data, target_activity_label, target_activity_name
    
    def _load_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """Load dataset by name"""
        dataset_name = dataset_name.lower()
        print(f"Loading dataset: {dataset_name}")
        
        if dataset_name in ["uci_adl_home_a", "uci_adl_home_b"]:
            return self._load_uci_adl_data(dataset_name)
        elif dataset_name in ["casas_aruba", "casas_cairo", "casas_kyoto", "casas_milan"]:
            return self._load_casas_data(dataset_name)
        else:
            raise ValueError(f"✗ Unsupported dataset: {dataset_name}")
    
    def _calculate_sensor_durations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate sensor duration based on ON/OFF status changes"""
        
        df['duration'] = 0.0
        
        # Group by sensor_id, scenario, instance, and subject to track state changes per sensor per session
        grouped = df.groupby(['sensor_id', 'scenario', 'instance', 'subject'])
        
        for (sensor_id, scenario, instance, subject), group in grouped:
            if len(group) < 2:
                continue
                
            # Sort by timestamp
            group_sorted = group.sort_values('timestamp')
            
            # Track ON/OFF state changes
            current_state = None
            on_timestamp = None
            on_event_idx = None
            
            for idx, row in group_sorted.iterrows():
                status = row['sensor_status']
                
                if status == 'ON' and current_state != 'ON':
                    # Sensor turned ON
                    on_timestamp = row['timestamp']
                    on_event_idx = idx
                    current_state = 'ON'
                    
                elif status == 'OFF' and current_state == 'ON':
                    # Sensor turned OFF - calculate duration
                    if on_timestamp is not None and on_event_idx is not None:
                        duration_seconds = (row['timestamp'] - on_timestamp).total_seconds()
                        
                        # Update duration only for the ON event
                        df.loc[on_event_idx, 'duration'] = duration_seconds
                    
                    current_state = 'OFF'
                    on_timestamp = None
                    on_event_idx = None
                    
                elif status == 'OFF' and current_state != 'OFF':
                    # Sensor was already OFF
                    current_state = 'OFF'
                    on_timestamp = None
                    on_event_idx = None
        
        # Update sensor_duration column
        df['sensor_duration'] = df['duration']
        
        # Keep only ON events
        df_filtered = df[(df['sensor_status'] == 'ON') & (df['duration'] >= 0)].copy()
        df_filtered = df_filtered.drop('duration', axis=1)
        
        return df_filtered
    
    def _load_uci_adl_data(self, dataset_name: str = "uci_adl_home_a") -> Tuple[List[pd.DataFrame], List[str], List[str]]:
        """UCI ADL Dataset loading"""
        activity_label = None
        activity_name = None
        data_path = self.config.uci_adl_data_path

        # Determine which dataset to load based on dataset_name (lowercase comparison)
        dataset_name_lower = dataset_name.lower()
        if dataset_name_lower == "uci_adl_home_a":
            adl_file = 'OrdonezA_ADLs.txt'
        elif dataset_name_lower == "uci_adl_home_b":
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
                        'sensor_context': sensor_row['place'],
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
    
    def _load_casas_data(self, dataset_name: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """Load CASAS Dataset (aruba, cairo, kyoto, milan)"""
        
        # CASAS sensor location mapping (based on typical smart home layouts)
        # Sensor types: M = Motion, D = Door, T = Temperature, AD = Analog
        casas_sensor_type_mapping = {
            'M': 'Motion',
            'D': 'Door',
            'T': 'Temperature',
            'AD': 'Analog'
        }
        
        # Determine dataset path and file
        dataset_name = dataset_name.lower()
        if dataset_name == "casas_aruba":
            data_path = self.config.casas_aruba_data_path
            data_file = "aruba.txt"
        elif dataset_name == "casas_cairo":
            data_path = self.config.casas_cairo_data_path
            data_file = "cairo.txt"
        elif dataset_name == "casas_kyoto":
            data_path = self.config.casas_kyoto_data_path
            data_file = "kyoto11.txt"
        elif dataset_name == "casas_milan":
            data_path = self.config.casas_milan_data_path
            data_file = "milan.txt"
        else:
            raise ValueError(f"✗ Unsupported CASAS dataset: {dataset_name}")
        
        file_path = os.path.join(data_path, data_file)
        
        if not os.path.exists(file_path):
            print(f"✗ CASAS data file not found: {file_path}")
            return pd.DataFrame(), np.array([]), []
        
        print(f"\nProcessing CASAS dataset: {dataset_name} from {file_path}")
        
        try:
            all_sensor_data = []
            current_activity = None
            activity_start_time = None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"  Loading {len(lines)} lines...")
            
            for line in tqdm(lines, desc=f"  Parsing {dataset_name}", leave=False):
                line = line.strip()
                if not line:
                    continue
                
                # Parse line based on dataset format
                parsed = self._parse_casas_line(line, dataset_name)
                if parsed is None:
                    continue
                
                timestamp, sensor_id, sensor_status, activity_info = parsed
                
                # Handle activity begin/end annotations
                if activity_info:
                    if 'begin' in activity_info.lower():
                        # Extract activity name (remove 'begin' suffix)
                        activity_name = activity_info.replace('begin', '').replace('Begin', '').strip()
                        # Clean up activity name (remove R1_, R2_ prefixes if present)
                        activity_name = self._clean_activity_name(activity_name)
                        current_activity = activity_name
                        activity_start_time = timestamp
                    elif 'end' in activity_info.lower():
                        current_activity = None
                        activity_start_time = None
                
                # Determine sensor type based on prefix
                sensor_type = 'Unknown'
                for prefix, type_name in casas_sensor_type_mapping.items():
                    if sensor_id.startswith(prefix):
                        sensor_type = type_name
                        break
                
                # Only include sensor events with known activity
                if current_activity and sensor_status in ['ON', 'OFF', 'OPEN', 'CLOSE']:
                    # Get sensor location and context from mapping
                    sensor_location, sensor_context = self._get_sensor_mapping(sensor_id, dataset_name)
                    
                    sensor_data = {
                        'timestamp': timestamp,
                        'sensor_id': sensor_id,
                        'sensor_location': sensor_location,
                        'sensor_type': sensor_type,
                        'sensor_context': sensor_context,
                        'sensor_status': sensor_status,
                        'activity': current_activity,
                        'sensor_duration': 0  # Will be calculated later
                    }
                    all_sensor_data.append(sensor_data)
            
            if len(all_sensor_data) == 0:
                print(f"  ✗ No valid sensor data found in {dataset_name}")
                return pd.DataFrame(), np.array([]), []
            
            # Convert to DataFrame
            df = pd.DataFrame(all_sensor_data)
            
            # Calculate sensor durations
            df = self._calculate_casas_sensor_durations(df)
            
            # Extract activity labels
            activity_label = np.array(df['activity'])
            activity_name = df['activity'].unique().tolist()
            
            print(f"\n✓ Created CASAS {dataset_name} dataset with {len(df)} sensor-activity pairs")
            print(f"  Sensor data sample:\n{df.head(3)}")
            print(f"  Activity names ({len(activity_name)}): {activity_name}")
            
            return df, activity_label, activity_name
            
        except Exception as e:
            print(f"✗ Error loading CASAS data {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(), np.array([]), []
    
    def _parse_casas_line(self, line: str, dataset_name: str) -> Optional[Tuple]:
        """Parse a single line from CASAS dataset"""
        try:
            # Different parsing for different CASAS datasets (lowercase comparison)
            dataset_name_lower = dataset_name.lower()
            if dataset_name_lower == "casas_aruba":
                # Format: 2010-11-04 00:03:50.209589 M003 ON Sleeping begin
                # Space-separated
                parts = line.split()
                if len(parts) < 4:
                    return None
                
                date_str = parts[0]
                time_str = parts[1]
                sensor_id = parts[2]
                sensor_status = parts[3]
                
                # Activity info (if present)
                activity_info = ' '.join(parts[4:]) if len(parts) > 4 else None
                
                timestamp = pd.to_datetime(f"{date_str} {time_str}", errors='coerce')
                
            elif dataset_name_lower == "casas_kyoto":
                # Format: 2009-02-02	07:15:16.575809	M35	ON	R1_Bed_to_Toilet begin
                # Tab-separated with separate date and time
                parts = line.split('\t')
                if len(parts) < 4:
                    return None
                
                date_str = parts[0].strip()
                time_str = parts[1].strip()
                sensor_id = parts[2].strip()
                sensor_status = parts[3].strip()
                
                # Activity info (if present)
                activity_info = parts[4].strip() if len(parts) > 4 else None
                
                timestamp = pd.to_datetime(f"{date_str} {time_str}", errors='coerce')
                
            else:  # cairo, milan
                # Format: 2009-06-10 00:00:00.024668	T003	19
                # or: 2009-10-16 00:01:04.000059	M017	ON
                # Tab-separated with combined date-time
                parts = line.split('\t')
                if len(parts) < 3:
                    return None
                
                datetime_str = parts[0].strip()
                sensor_id = parts[1].strip()
                sensor_status = parts[2].strip()
                
                # Activity info (if present)
                activity_info = parts[3].strip() if len(parts) > 3 else None
                
                timestamp = pd.to_datetime(datetime_str, errors='coerce')
            
            if pd.isna(timestamp):
                return None
            
            return timestamp, sensor_id, sensor_status, activity_info
            
        except Exception as e:
            return None
    
    def _clean_activity_name(self, activity_name: str) -> str:
        """Clean activity name by removing prefixes and standardizing format"""
        # Remove R1_, R2_ prefixes (for multi-resident datasets like kyoto)
        import re
        activity_name = re.sub(r'^R\d+[_\s]*', '', activity_name)
        
        # Replace underscores with spaces
        activity_name = activity_name.replace('_', ' ')
        
        # Capitalize first letter and strip
        activity_name = activity_name.strip().capitalize()
        
        # Apply unified activity mapping
        activity_name = self._map_activity_to_category(activity_name)
        
        return activity_name
    
    def _map_activity_to_category(self, activity_name: str) -> str:
        """Map activity name to unified category"""
        normalized = activity_name.lower().replace(' ', '').replace('_', '')
        
        # Unified activity mapping
        activity_mapping = {
            # Relax
            'relax': 'Relax',
            'watchtv': 'Relax',
            'read': 'Relax',
            'sparetime/tv': 'Relax',
            
            # Cook
            'mealpreparation': 'Cook',
            'kitchenactivity': 'Cook',
            
            # Eat
            'eating': 'Eat',
            'breakfast': 'Eat',
            'lunch': 'Eat',
            'dinner': 'Eat',
            'diningroomactivity': 'Eat',
            'snack': 'Eat',

            # Work
            'work': 'Work',
            'washdishes': 'Work',
            'housekeeping': 'Work',
            'workinoffice': 'Work',
            'laundry': 'Work',
            'deskactivity': 'Work',
            'chores': 'Work',
            
            # Bathing
            'bathing': 'Bathing',
            'guestbathroom': 'Bathing',
            'masterbathroom': 'Bathing',
            'showering': 'Bathing',
            
            # PersonalHygiene
            'personalhygiene': 'Personal Hygiene',
            'toileing': 'Personal Hygiene',
            'grooming': 'Personal Hygiene',

            # Sleep
            'sleeping': 'Sleep',
            'sleep': 'Sleep',
            'sleepingnotinbed': 'Sleep',
            
            # BedtoToilet
            'bedtotoilet': 'Bed to Toilet',
            'bedtoilettransition': 'Bed to Toilet',
            
            # TakeMedicine
            'takemedicine': 'Take Medicine',
            'morningmeds': 'Take Medicine',
            'eveningmeds': 'Take Medicine',
            
            # Enter Home
            'enterhome': 'Enter Home',
            
            # Leave Home
            'leavehome': 'Leave Home',
            'leaving': 'Leave Home',

            # Other
            'resperate': 'Other',
            'nightwandering': 'Other',
            'wake': 'Other',
            'wanderinginroom': 'Other',
            'masterbedroomactivity': 'Other',
            'meditate': 'Other',
            'other': 'Other'
        }
        
        if normalized in activity_mapping:
            return activity_mapping[normalized]
        
        # If no exact match, return original (capitalized)
        return activity_name
    
    def _get_sensor_mapping(self, sensor_id: str, dataset_name: str) -> Tuple[str, str]:
        """Get sensor context and location based on sensor ID and dataset
        
        Returns:
            Tuple[str, str]: (sensor_location, sensor_context)
        """
        dataset_name = dataset_name.lower()

        # CASAS mapping: {sensor_id: [location, context]}
        if dataset_name == "casas_aruba":
            # Aruba: Single-resident home
            # Single story home with living space, dining space, kitchen, office, 2 bedrooms, 2 bathrooms, and closet
            aruba_mapping = {
                "M001": ["Bedroom 1", "in first bedroom"],
                "M002": ["Bedroom 1", "on bed of first bedroom"],
                "M003": ["Bedroom 1", "on bed of first bedroom"],
                "M004": ["Bathroom 1", "between first bedroom and first bathroom"],
                "M005": ["Bedroom 1", "in first bedroom"],
                "M006": ["Bedroom 1", "on first bedroom door"],
                "M007": ["Bedroom 1", "in first bedroom"],
                "M008": ["Corridor near bedroom 1", "in home entrance corridor"],
                "M009": ["Living room", "between living room and home entrance corridor"],
                "M010": ["Living room", "between living room and home entrance corridor"],
                "M011": ["Corridor near bedroom 1", "near front door"],
                "M012": ["Living room", "in living room"],
                "M013": ["Living room", "between dining area and living room"],
                "M014": ["Dining area", "in dining area"],
                "M015": ["Kitchen", "in kitchen"],
                "M016": ["Kitchen", "near back door"],
                "M017": ["Kitchen", "in kitchen"],
                "M018": ["Kitchen", "between kitchen and dining area"],
                "M019": ["Kitchen", "in kitchen"],
                "M020": ["Living room", "in living room"],
                "M021": ["Corridor near bathroom 2", "in corridor between second bathroom and living room"],
                "M022": ["Corridor near bathroom 2", "in corridor between second bathroom and second bedroom"],
                "M023": ["Bedroom 2", "on second bedroom door"],
                "M024": ["Bedroom 2", "in second bedroom"],
                "M025": ["Office", "in office"],
                "M026": ["Office", "on office desk and chair"],
                "M027": ["Office", "in office"],
                "M028": ["Office", "on office door between office and corridor near second bathroom"],
                "M029": ["Bathroom 2", "between corridor and second bathroom"],
                "M030": ["Garage", "in corridor between garage door and second bathroom"],
                "M031": ["Corridor near bathroom 2", "in corridor near bathroom 2"],

                "D001": ["Front door", "in home entrance corridor"],
                "D002": ["Back door", "between kitchen and back door"],
                "D003": ["Corridor near bathroom 2", "in corridor near second bathroom"],
                "D004": ["Garage door", "on garage door"],

                "T001": ["Bedroom 1", "in first bedroom"],
                "T002": ["Living room", "in living room"],
                "T003": ["Kitchen", "in kitchen"],
                "T004": ["Corridor near bathroom 2", "in corridor between second bathroom and dining area"],
                "T005": ["Office", "in office"]
            }
            if sensor_id in aruba_mapping:
                return aruba_mapping[sensor_id][0], aruba_mapping[sensor_id][1]
            return "Unknown", "Unknown"
            
        elif dataset_name == "casas_cairo":
            # Cairo: Two-resident home (R1, R2)
            # Three story home with living space, dining space, kitchen, 2 bedrooms, office, laundry room, and garage room
            cairo_mapping = {
                "M001": ["Office", "in office"],
                "M002": ["Corridor near bedroom", "between bedroom, guest room and office"],
                "M003": ["Corridor near bedroom", "between office and bedroom"],
                "M004": ["Guest room", "in guest room"],
                "M005": ["Bedroom", "in bedroom"],
                "M006": ["Bedroom", "on door of bedroom"],
                "M007": ["Bedroom", "near bathroom in bedroom"],
                "M008": ["Bedroom", "on bed in bedroom"],
                "M009": ["Bedroom", "on bed in bedroom"],
                "M010": ["Corridor near bedroom", "close to stairs in corridor near bedroom"],
                "M011": ["Living room", "near bottom of stairs in living room"],
                "M012": ["Kitchen", "in kitchen"],
                "M013": ["Living room", "near couch in living room"],
                "M014": ["Living room", "near stairs in living room"],
                "M015": ["Entrance", "near outside door in living room"],
                "M016": ["Living room", "in living room"],
                "M017": ["Living room", "near couch in living room"],
                "M018": ["Living room", "in living room"],
                "M019": ["Kitchen", "near dining area in kitchen"],
                "M020": ["Dining area", "in dining area"],
                "M021": ["Kitchen", "near medicine cabinet in kitchen"],
                "M022": ["Kitchen", "in kitchen"],
                "M023": ["Living room", "in living room"],
                "M024": ["Kitchen", "in kitchen"],
                "M025": ["Other room", "in other room"],
                "M026": ["Laundary room", "in laundary room"],
                "M027": ["Laundary room", "near garage door in laundary room"],

                "T001": ["Bedroom", "near bathroom in bedroom"],
                "T002": ["Office", "in office"],
                "T003": ["Living room", "near stairs in living room"],
                "T004": ["Kitchen", "near medicine cabinet in kitchen"],
                "T005": ["Living room", "in living room"]
            }
            if sensor_id in cairo_mapping:
                return cairo_mapping[sensor_id][0], cairo_mapping[sensor_id][1]
            return "Unknown", "Unknown"
            
        elif dataset_name == "casas_kyoto":
            # Kyoto: Multi-resident dataset (R1, R2)
            kyoto_mapping = {
                
            }
            if sensor_id in kyoto_mapping:
                return kyoto_mapping[sensor_id][0], kyoto_mapping[sensor_id][1]
            return "Unknown", "Unknown"
            
        elif dataset_name == "casas_milan":
            # Milan: Single-resident home
            # Single story home with living space, dining space, kitchen, workspace/TV room, 2 bedrooms, 2 bathrooms, and garage
            milan_mapping = {
                "M001": ["Entrance", "near home entrance"],
                "M002": ["Entrance", "near home entrance towards living room"],
                "M003": ["Dining area", "on table in dining area"],
                "M004": ["Living room", "on sofa in living room"],
                "M005": ["Living room", "in living room near slider door"],
                "M006": ["Workspace/TV room", "between living room and workspace/TV room"],
                "M007": ["Workspace/TV room", "near desk in workspace/TV room"],
                "M008": ["Workspace/TV room", "on sofa in workspace/TV room"],
                "M009": ["Corridor near bathroom 1", "in corridor near first bathroom close to the washer and dryer"],
                "M010": ["Corridor near kitchen", "in corridor between dining area and kitchen"],
                "M011": ["Corridor near kitchen", "in corridor between second bathroom and kitchen"],
                "M012": ["Kitchen", "between dining area and kitchen"],
                "M013": ["Bathroom 1", "near sink in first bathroom"],
                "M014": ["Kitchen", "near door in kitchen"],
                "M015": ["Kitchen", "near fridge in the kitchen"],
                "M016": ["Kitchen", "between corridor and medicine cabinet in the kitchen"],
                "M017": ["Bathroom 2", "near sink in second bathroom"],
                "M018": ["Bathroom 2", "near toilet/shower in second bathroom"],
                "M019": ["Corridor near bathroom 1", "in corridor between workspace/TV room and first bathroom"],
                "M020": ["Bedroom 1", "between first bedroom and first bathroom"],
                "M021": ["Bedroom 1", "on bed in first bedroom"],
                "M022": ["Kitchen", "near stove in kitchen"],
                "M023": ["Kitchen", "in kitchen"],
                "M024": ["Bedroom 2", "in second bedroom"],
                "M025": ["Bathroom 1", "between first bathroom and walk-in closet"],
                "M026": ["Workspace_TV room", "in workspace/TV room"],
                "M027": ["Living room", "in living room"],
                "M028": ["Bedroom 1", "in first bedroom"],
                "D001": ["Entrance", "on home entrance door"],
                "D002": ["Entrance", "on coat cabinet near home entrance door"],
                "D003": ["Kitchen", "on kitchen door"],
                "T001": ["Kitchen", "in kitchen near stove"],
                "T002": ["Corridor near bathroom 2", "in corridor near second bathroom"]
            }
            if sensor_id in milan_mapping:
                return milan_mapping[sensor_id][0], milan_mapping[sensor_id][1]
            return "Unknown", "Unknown"
        
        # Default fallback
        return "Unknown", "Unknown"
    
    def _calculate_casas_sensor_durations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate sensor duration based on ON/OFF status changes for CASAS data"""
        
        df['duration'] = 0.0
        
        # Group by sensor_id to track state changes per sensor
        grouped = df.groupby('sensor_id')
        
        for sensor_id, group in grouped:
            if len(group) < 2:
                continue
            
            # Sort by timestamp
            group_sorted = group.sort_values('timestamp')
            
            # Track ON/OFF state changes
            current_state = None
            on_timestamp = None
            on_event_idx = None
            
            for idx, row in group_sorted.iterrows():
                status = row['sensor_status']
                
                if status in ['ON', 'OPEN'] and current_state not in ['ON', 'OPEN']:
                    # Sensor turned ON/OPEN
                    on_timestamp = row['timestamp']
                    on_event_idx = idx
                    current_state = status
                    
                elif status in ['OFF', 'CLOSE'] and current_state in ['ON', 'OPEN']:
                    # Sensor turned OFF/CLOSE - calculate duration
                    if on_timestamp is not None and on_event_idx is not None:
                        duration_seconds = (row['timestamp'] - on_timestamp).total_seconds()
                        
                        # Update duration only for the ON event
                        df.loc[on_event_idx, 'duration'] = duration_seconds
                    
                    current_state = status
                    on_timestamp = None
                    on_event_idx = None
                    
                elif status in ['OFF', 'CLOSE'] and current_state not in ['ON', 'OPEN']:
                    # Sensor was already OFF/CLOSE
                    current_state = status
                    on_timestamp = None
                    on_event_idx = None
        
        # Update sensor_duration column
        df['sensor_duration'] = df['duration']
        
        # Keep only ON/OPEN events with calculated duration
        df_filtered = df[(df['sensor_status'].isin(['ON', 'OPEN'])) & (df['duration'] >= 0)].copy()
        df_filtered = df_filtered.drop('duration', axis=1)
        
        return df_filtered
    
    def get_data(self):
        return self.source_sensor_data, self.source_activity_label, self.source_activity_name, self.target_sensor_data, self.target_activity_label, self.target_activity_name

def load_sensor_data(config: SemanticHARConfig, use_event_based: bool = True) -> Dict[str, Dict[str, List[pd.DataFrame]]]:
    """Load sensor data as DataFrames and split with event-based or time-based windows"""

    dataset_loader = SensorDataset(config)
    source_sensor_data, _, source_activity_name, target_sensor_data, _, target_activity_name = dataset_loader.get_data()

    split_data = {
        'train': {},
        'val': {},
        'test': {}
    }
    
    # Process source dataset (train + val)
    print(f"\nProcessing source dataset: {config.source_dataset}")
    
    if source_sensor_data is None or len(source_sensor_data) == 0:
        print(f"  ✗ No source sensor data available")
        return split_data
    
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
    
    if target_sensor_data is None or len(target_sensor_data) == 0:
        print(f"  ✗ No target sensor data available")
        return split_data
    
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
    _save_windows(split_data, source_activity_name, target_activity_name, config, use_event_based)
    
    return split_data


def _create_event_based_windows(df: pd.DataFrame) -> List[pd.DataFrame]:
    """Create event-based windows"""
    
    if df is None or len(df) == 0:
        print("    Warning: Empty DataFrame provided to _create_event_based_windows")
        return []
    
    # Check if timestamp column exists
    if 'timestamp' not in df.columns:
        print(f"    Error: 'timestamp' column not found in DataFrame. Available columns: {df.columns.tolist()}")
        return []
    
    # Convert timestamp to datetime if it's not already
    if df['timestamp'].dtype in ['int64', 'int32', 'float64', 'float32']:
        # Assume timestamp is in milliseconds (UNIX timestamp)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
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
    
    if df is None or len(df) == 0:
        print("    Warning: Empty DataFrame provided to _create_time_based_windows")
        return []
    
    # Check if timestamp column exists
    if 'timestamp' not in df.columns:
        print(f"    Error: 'timestamp' column not found in DataFrame. Available columns: {df.columns.tolist()}")
        return []
    
    # Convert timestamp to datetime if it's not already
    if df['timestamp'].dtype in ['int64', 'int32', 'float64', 'float32']:
        # Assume timestamp is in milliseconds (UNIX timestamp)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
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


def _save_windows(split_data: Dict, source_activity_name: List, target_activity_name: List, config, use_event_based: bool = True) -> str:
    """Save windows to JSON file"""
    
    # Base generation info
    generation_info = {
        'timestamp': datetime.now().isoformat(),
        'source_dataset': config.source_dataset,
        'target_dataset': config.target_dataset,
        'source_activity_label': source_activity_name,
        'target_activity_label': target_activity_name
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
                    contexts = window['sensor_context'].unique().tolist() if 'sensor_context' in window.columns else []
                    
                    window_info = {
                        'window_id': f'{dataset}_{split_name}_{i+1}',
                        'window_start': window_start,
                        'window_end': window_end,
                        'window_duration': window_duration,
                        'activity': activity,
                        'sensor_types': sensor_types,
                        'locations': locations,
                        'contexts': contexts
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
