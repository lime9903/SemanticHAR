"""
Using LLM to generate semantic interpretations for sensor data
"""
import os
import sys
import json
import time
import openai
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SemanticHARConfig


class SemanticGenerator:
    """LLM to generate semantic interpretations for sensor data"""
    
    def __init__(self, config: SemanticHARConfig):
        self.config = config
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Semantic interpretations repository
        self.sensor_interpretations = {}
        self.activity_interpretations = {}
        
        print(f"⨠ SemanticGenerator initialized")
        print(f"   Device: {config.device}")
        from config import OPENAI_API_KEY
        print(f"   OpenAI API: {'✓' if OPENAI_API_KEY else '✗'}")
    

    def generate_sensor_interpretation(self, window_data: Dict) -> str:
        """Generate semantic interpretations for sensor data"""
        
        system_prompt, user_prompt = self._create_sensor_prompt(window_data)
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature_llm
            )
            
            interpretation = response.choices[0].message.content.strip()
            return interpretation
            
        except Exception as e:
            print(f"LLM call error: {e}")
            return None
    

    def generate_activity_interpretation(self, activity: str) -> str:
        """Generate semantic interpretations for activity labels"""
        
        system_prompt, user_prompt = self._create_activity_prompt(activity)
        
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature_llm
            )
            
            interpretation = response.choices[0].message.content.strip()
            return interpretation
            
        except Exception as e:
            print(f"LLM call error: {e}")
            return None


    def generate_interpretations(self, config: SemanticHARConfig) -> str:
        """
        Generate semantic interpretations
        
        Args:
            config: configuration object
            splits: data splits to process (train, val, test)
            
        Returns:
            path to the generated interpretations file
        """
        
        # Load windows data
        windows_file = config.windows_file
        if not os.path.exists(windows_file):
            print(f"✗ Windows file not found: {windows_file}")
            print(f"  Please run 'python main.py --mode train' first to generate the windows")
            return None
            
        with open(windows_file, 'r', encoding='utf-8') as f:
            windows_data = json.load(f)
        
        # Initialize result structure
        results = {
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'source_dataset': config.source_dataset,
                'target_dataset': config.target_dataset
            },
            'sensor_interpretations': {
                'train': {},
                'val': {},
                'test': {}
            },
            'activity_interpretations': {},
            'statistics': {}
        }
        
        if not config.use_event_based:
            results['generation_info'].update({
                'window_size_seconds': config.window_size_seconds,
                'overlap_ratio': config.overlap_ratio
            })
        
        # Process each split
        total_windows_processed = 0
        total_interpretations_generated = 0

        for split in windows_data['windows']:
            print(f"\n⨠ Processing {split} split...")
            
            windows = windows_data['windows'][split]
            if len(windows) == 0:
                print(f"    ✗  No {split} windows available")
                continue
            
            print(f"  ⨠ Processing {len(windows)} windows")

            split_results = {}
            successful_interpretations = 0
            
            # Process each window
            for i, window_data in enumerate(tqdm(windows, desc=f"    {split}")):
                try:
                    window_id = window_data['window_id']
                    # Generate sensor interpretation
                    interpretation = self.generate_sensor_interpretation(window_data)

                    split_results[f"window_{i+1}"] = {
                        'interpretation': interpretation,
                        'activity': window_data.get('activity', 'Unknown'),
                        'window_id': window_id  # To find corresponding window data for sensor encoder training
                    }

                    successful_interpretations += 1
                    total_interpretations_generated += 1
                    
                except Exception as e:
                    print(f"    ✗ Error processing window {i+1}: {e}")
                    activity = window_data.get('activity', 'Unknown') if 'window_data' in locals() else 'Unknown'
                    
                    split_results[f"window_{i+1}"] = {
                        'error': str(e),
                        'activity': activity
                    }
            
            results['sensor_interpretations'][split] = split_results
            total_windows_processed += len(windows)
            
            print(f"    ✓ {split}: {successful_interpretations}/{len(windows)} successful")
        
        # Generate activity interpretations
        print(f"\n⨠ Generating activity interpretations...")
        unique_activities = set()
        for split in results['sensor_interpretations']:
            for window_id, window_data in results['sensor_interpretations'][split].items():
                if 'activity' in window_data:
                    unique_activities.add(window_data['activity'])
        
        unique_activities = list(unique_activities)
        print(f"⨠ Processing {len(unique_activities)} unique activities")
        
        activity_interpretations = {}
        for activity in tqdm(unique_activities, desc="  Activities"):
            try:
                interpretation = self.generate_activity_interpretation(activity)
                activity_interpretations[activity] = {
                    'interpretation': interpretation,
                    'activity': activity
                }
            except Exception as e:
                print(f"    ✗ Error processing activity {activity}: {e}")
                activity_interpretations[activity] = {
                    'error': str(e),
                    'activity': activity
                }
        
        results['activity_interpretations'] = activity_interpretations
        
        # Generate statistics
        results['statistics'] = {
            'total_windows_processed': total_windows_processed,
            'total_interpretations_generated': total_interpretations_generated,
            'successful_sensor_interpretations': sum(len(window_data) for split in results['sensor_interpretations'] for window_data in results['sensor_interpretations'][split].values()),
            'successful_activity_interpretations': len(activity_interpretations),
            'unique_activities': len(unique_activities),
        }
        
        # Save results
        os.makedirs(os.path.dirname(config.semantic_interpretations_file), exist_ok=True)
        
        with open(config.semantic_interpretations_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Result summary
        print("\n" + "-" * 64)
        print("GENERATION SUMMARY")
        print(f"  - Total windows processed: {total_windows_processed}")
        print(f"  - Total interpretations generated: {total_interpretations_generated}")
        print(f"  - Successful sensor interpretations: {results['statistics']['successful_sensor_interpretations']}")
        print(f"  - Successful activity interpretations: {results['statistics']['successful_activity_interpretations']}")
        print(f"  - Unique activities: {results['statistics']['unique_activities']}")
        print(f"  - Output file: {config.semantic_interpretations_file}")
        print("-" * 64 + "\n")
        
        return config.semantic_interpretations_file
    

    def load_interpretations(self, interpretations_file: str) -> Dict:
        """Load existing interpretations file"""
        print(f"⨠ Loading interpretations from: {interpretations_file}")
        
        with open(interpretations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ Interpretations loaded successfully")
        print(f"   - Splits: {len(data.get('sensor_interpretations', {}))}")
        print(f"   - Activities: {len(data.get('activity_interpretations', {}))}")
        
        return data

    
    def _create_sensor_prompt(self, window_data: Dict) -> Tuple[str, str]:
        """Create prompt for ambient sensor data interpretation"""
        # Remove 'window_id' key in dictionary
        window_data.pop('window_id', None)

        # System prompt: General instructions and knowledge
        system_prompt = f"""You are an expert in analyzing ambient sensor data for human activity recognition. You specialize in interpreting sensor patterns from smart home environments.

**Your Expertise**:
- Analyzing PIR, Magnetic, Pressure, Flush, and Electric sensor data
- Understanding human activity patterns in smart home environments
- Providing detailed, step-by-step analysis with clear reasoning
- Categorizing sensor patterns into meaningful classifications

**Ambient Sensor Knowledge**:
- PIR Sensors: Detect motion and presence in specific areas, showing activity intensity through frequency and duration
- Magnetic Sensors: Detect door/window openings and closings, indicating room transitions and access patterns
- Pressure Sensors: Detect weight distribution and contact, showing seating, lying, or standing patterns
- Flush Sensors: Detect toilet usage patterns, indicating bathroom activities
- Electric Sensors: Detect appliance usage and power consumption, showing device interaction patterns

**Activity-Specific Knowledge**:
{self._get_activity_knowledge(window_data['activity'])}

**Analysis Framework**:
You will analyze sensor data using amplitude analysis, frequency analysis, time series analysis, and statistical measures. You will categorize patterns into 6 categories: Sensor Activity Level, Temporal Pattern, Location Distribution, Sensor Type Pattern, Activity Intensity, and Duration Pattern."""
        
        # User prompt: semantic interpretation focus
        user_prompt = f"""## Data Introduction

This is a {window_data['window_duration']} seconds time window of ambient sensor data from a smart home environment. The first element is the start time of window, the second element is the end time of window, the third element is the duration of window, the fourth element is the activity, the fifth element is the sensor type, the sixth element is the sensor location, the seventh element is the sensor place. The ambient sensor reading may be in one of the following states: [Sleeping, Toileting, Showering, Breakfast, Lunch, Dinner, Grooming, Spare_Time/TV, Leaving, Snack].

## Task Introduction: Semantic Interpretation Generation
Generate a comprehensive semantic interpretation of the sensor data that describes the human activity patterns. Focus on:

1. **Activity Description**: What the person was likely doing during this time window
2. **Sensor Pattern Analysis**: How the sensor events relate to the activity
3. **Behavioral Insights**: What the sensor data reveals about user behavior
4. **Temporal Characteristics**: Timing patterns and activity duration
5. **Spatial Context**: Location-based activity patterns
6. **Confidence Assessment**: Confidence assessment of the interpretation

**Output Format**:
You MUST provide your analysis in the following exact format with these exact headers:

Activity Description: [Describe what the person was likely doing during this time window]

Sensor Pattern Analysis: [Analyze how the sensor events relate to the activity]

Behavioral Insights: [What the sensor data reveals about user behavior]

Temporal Characteristics: [Timing patterns and activity duration analysis]

Spatial Context: [Location-based activity patterns]

Confidence Assessment: [Your confidence level in this interpretation (High/Medium/Low) with reasoning]

**IMPORTANT FORMATTING REQUIREMENTS**:
- Use EXACTLY the headers shown above (copy them exactly)
- Each section must start with 'Header Name':
- Provide 2-3 sentences for each section
- Be specific and detailed in your analysis
- Consider the activity context
- Provide reasoning for your interpretations
- Maintain scientific accuracy

**EXAMPLE OUTPUT FORMAT**:
**Activity Description**: The person was likely taking a shower in the bathroom, as indicated by the PIR sensor detecting movement in the shower area for an extended duration.

**Sensor Pattern Analysis**: The PIR sensor in the shower location was triggered for 298 seconds, indicating continuous movement and activity in the shower area, which is consistent with showering behavior.

**Behavioral Insights**: The extended duration suggests the person took a thorough shower, and the single sensor activation indicates focused activity in one location without room transitions.

**Temporal Characteristics**: The activity lasted for 298 seconds (approximately 5 minutes), which is a typical duration for a shower, with continuous sensor activation throughout.

**Spatial Context**: The activity was confined to the bathroom area, specifically the shower location, with no movement to other rooms during this time window.

**Confidence Assessment**: High confidence - the sensor pattern strongly indicates showering activity based on location (shower), duration (typical shower time), and sensor type (PIR detecting movement).

## Sensor Data:

{json.dumps(window_data, indent=2)}"""
     
        return system_prompt, user_prompt
    

    def _create_activity_prompt(self, activity: str) -> Tuple[str, str]:
        """create prompt for ambient sensor activity label interpretation"""
        
        # System prompt: General instructions and expertise
        system_prompt = """You are an expert in analyzing human activities in smart home environments. You specialize in generating detailed descriptions of activities based on ambient sensor patterns.

**Your Expertise**:
- Understanding human activity patterns in smart home environments
- Analyzing ambient sensor data patterns (PIR, Magnetic, Pressure, Flush, Electric)
- Describing activity characteristics from sensor perspectives
- Identifying environmental context and spatial patterns of activities

**Ambient Sensor Knowledge**:
- **PIR Sensors**: Detect motion and presence, indicating movement patterns and activity intensity
- **Magnetic Sensors**: Detect door/window/cabinet openings, indicating access patterns and room transitions
- **Pressure Sensors**: Detect weight distribution and contact, indicating seating, lying, or standing patterns
- **Flush Sensors**: Detect toilet usage patterns, indicating bathroom activities
- **Electric Sensors**: Detect appliance usage, indicating device interaction patterns

**Analysis Framework**:
You will generate descriptions covering three key aspects: General Description, Ambient Sensor Patterns, and Environmental Context. Focus on how ambient sensors would detect and characterize the activity."""
        
        # User prompt: Specific task
        user_prompt = f"""You need to generate a detailed description for the given activity: {activity}

The description should include three aspects:

## 1. General Description
- Provide an overview of the activity, including its purpose
- Common settings where it is performed in smart home environments
- Frequency and importance in daily life
- Duration and typical timing of the activity
- Any other relevant details about the activity

## 2. Sensor Patterns Detected by Sensors
Describe the typical patterns detected by ambient sensors during this activity:

### PIR Sensor Pattern:
- Strength and frequency of movement detection
- Change of position in the activity space
- Duration and periodicity of the activity
- Spatial coverage and movement intensity

### Magnetic Sensor Pattern:
- Door and window opening/closing patterns
- Cabinet and drawer access patterns
- Frequency and timing of magnetic sensor triggers
- Room transition patterns

### Pressure Sensor Pattern:
- Weight distribution patterns on beds, chairs, or floors
- Contact and seating patterns
- Duration and intensity of pressure changes
- Posture and positioning indicators

### Electric Sensor Pattern:
- Appliance usage patterns (microwave, toaster, TV, etc.)
- Power consumption patterns
- Device interaction timing and frequency
- Electrical device activation sequences

### Flush Sensor Pattern (if applicable):
- Toilet usage patterns
- Frequency and timing of flush events
- Bathroom activity indicators

## 3. Environmental Context and Spatial Patterns
Identify and describe the environmental context and spatial patterns involved in the activity:
- Primary spaces and rooms where the activity takes place
- Environmental objects and devices typically involved
- Spatial movement patterns between different areas
- Room-to-room transitions and access patterns
- Environmental conditions and settings

Each aspect should be described in detail and technically, focusing on how ambient sensors would detect and characterize the activity patterns."""
        
        return system_prompt, user_prompt
    

    def _get_activity_knowledge(self, activity: str) -> str:
        """Get activity-specific knowledge for system prompt"""
        activity_knowledge = {
            'Sleeping': "Sleeping: during sleeping, the ambient sensors may detect minimal activity with occasional PIR sensor triggers for position changes, pressure sensors on beds showing consistent weight distribution, and very low overall sensor activity.",
            'Toileting': "Toileting: during toileting, the ambient sensors may detect a sequence of PIR sensors for movement, magnetic sensors for door opening/closing, pressure sensors on toilet seats, and flush sensors for toilet usage, typically showing moderate to high sensor activity.",
            'Showering': "Showering: during showering, the ambient sensors may detect PIR sensors for movement in shower area, pressure sensors for water flow patterns, and temperature-related sensor changes, typically showing moderate sensor activity concentrated in bathroom area.",
            'Breakfast': "Breakfast: during breakfast, the ambient sensors may detect PIR sensors in kitchen and dining areas, magnetic sensors for fridge/cabinet openings, electric sensors for appliance usage, and pressure sensors for seating, typically showing high sensor activity across multiple locations.",
            'Grooming': "Grooming: during grooming, the ambient sensors may detect PIR sensors for bathroom movement, pressure sensors for sink usage, and magnetic sensors for cabinet/drawer openings, typically showing moderate sensor activity in bathroom area.",
            'Spare_Time/TV': "Spare_Time/TV: during leisure time, the ambient sensors may detect PIR sensors showing consistent presence in seating areas, minimal movement patterns, and low overall sensor activity with occasional electric sensor triggers for remote controls.",
            'Lunch': "Lunch: similar to breakfast but may involve different food preparation methods, timing patterns, and sensor activity levels depending on meal complexity.",
            'Dinner': "Dinner: dinner preparation may involve more complex cooking activities, multiple kitchen appliances, longer preparation times, and higher sensor activity across kitchen and dining areas."
        }
        return activity_knowledge.get(activity, f"General knowledge about {activity} activities in smart home environments.")
