"""Sensor detection system for ASW simulation."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import tomllib
from pathlib import Path


@dataclass
class Sensor:
    """Represents a single sensor with detection capabilities."""
    time: float  # When sensor becomes active (minutes)
    range: float  # 50% detection likelihood range (yards)
    rate: float  # Detections per minute
    x: float  # Sensor position (yards)
    y: float  # Sensor position (yards)
    name: Optional[str] = None
    
    def __post_init__(self):
        if self.name is None:
            self.name = f"Sensor_{self.x:.0f}_{self.y:.0f}"


@dataclass
class DetectionModel:
    """Gaussian detection model configuration."""
    type: str = "gaussian"
    range_decay: float = 0.5  # Detection probability at range distance
    noise_floor: float = 0.01  # Minimum detection probability


@dataclass
class SearchPattern:
    """Complete search pattern configuration."""
    name: str
    description: str
    sensors: List[Sensor]
    detection_model: DetectionModel


class DetectionEngine:
    """Engine for calculating detection probabilities and statistics."""
    
    def __init__(self, search_pattern: SearchPattern):
        self.search_pattern = search_pattern
        self.detection_model = search_pattern.detection_model
        
    def calculate_detection_probability(self, sensor: Sensor, target_positions: np.ndarray) -> np.ndarray:
        """Calculate detection probability for each target given a sensor.
        
        Args:
            sensor: The sensor to evaluate
            target_positions: (N, 2) array of target [x, y] positions
            
        Returns:
            (N,) array of detection probabilities for each target
        """
        # Calculate distances from sensor to each target
        distances = np.sqrt(
            (target_positions[:, 0] - sensor.x)**2 + 
            (target_positions[:, 1] - sensor.y)**2
        )
        
        # Gaussian detection model: P(detect) = exp(-(d/range)^2)
        # At range distance, probability = range_decay
        # So: range_decay = exp(-(range/range)^2) = exp(-1)
        # Therefore: P(detect) = exp(-(d/range)^2)
        probabilities = np.exp(-(distances / sensor.range)**2)
        
        # Scale by detection rate to get realistic probabilities
        # The rate represents the maximum possible detections per minute
        # So we scale the probability by rate / (number of targets)
        max_probability = sensor.rate / len(target_positions)
        probabilities = probabilities * max_probability
        
        # Apply noise floor
        probabilities = np.maximum(probabilities, self.detection_model.noise_floor)
        
        return probabilities
    
    def evaluate_sensor_at_time(self, sensor: Sensor, target_positions: np.ndarray, 
                               dt: float) -> Dict[str, Any]:
        """Evaluate a single sensor's performance at a given time.
        
        Args:
            sensor: The sensor to evaluate
            target_positions: (N, 2) array of target positions
            dt: Time step size (minutes)
            
        Returns:
            Dictionary with detection statistics
        """
        # Check if sensor is active at this time
        if sensor.time > 0:  # Sensor not yet active
            return {
                'detections': 0,
                'detection_rate': 0.0,
                'coverage': 0.0,
                'active': False
            }
        
        # Calculate detection probabilities
        probabilities = self.calculate_detection_probability(sensor, target_positions)
        
        # Expected number of detections per time step
        expected_detections = np.sum(probabilities) * dt
        
        # Coverage: fraction of targets with detection probability > threshold
        # Use a threshold based on the maximum possible probability
        max_probability = sensor.rate / len(target_positions)
        coverage_threshold = max_probability * 0.1  # 10% of max probability
        coverage = np.mean(probabilities > coverage_threshold)
        
        return {
            'detections': expected_detections,
            'detection_rate': expected_detections / dt if dt > 0 else 0.0,
            'coverage': coverage,
            'active': True,
            'probabilities': probabilities
        }
    
    def evaluate_pattern_at_time(self, target_positions: np.ndarray, 
                                current_time: float, dt: float) -> Dict[str, Any]:
        """Evaluate the entire search pattern at a given time.
        
        Args:
            target_positions: (N, 2) array of target positions
            current_time: Current simulation time (minutes)
            dt: Time step size (minutes)
            
        Returns:
            Dictionary with pattern-wide statistics
        """
        sensor_results = {}
        total_detections = 0.0
        total_coverage = 0.0
        active_sensors = 0
        
        for sensor in self.search_pattern.sensors:
            # Adjust sensor time relative to current simulation time
            adjusted_sensor = Sensor(
                time=sensor.time - current_time,  # Time until activation
                range=sensor.range,
                rate=sensor.rate,
                x=sensor.x,
                y=sensor.y,
                name=sensor.name
            )
            
            result = self.evaluate_sensor_at_time(adjusted_sensor, target_positions, dt)
            sensor_results[sensor.name] = result
            
            if result['active']:
                total_detections += result['detections']
                total_coverage += result['coverage']
                active_sensors += 1
        
        # Pattern-wide statistics
        avg_coverage = total_coverage / max(active_sensors, 1)
        
        return {
            'sensor_results': sensor_results,
            'total_detections': total_detections,
            'total_detection_rate': total_detections / dt,
            'average_coverage': avg_coverage,
            'active_sensors': active_sensors
        }


def load_search_pattern(config_path: Path) -> SearchPattern:
    """Load search pattern configuration from TOML file.
    
    Args:
        config_path: Path to TOML configuration file
        
    Returns:
        SearchPattern object
    """
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    
    # Parse sensors
    sensors = []
    for sensor_config in config.get('sensor', []):
        sensor = Sensor(
            time=sensor_config['time'],
            range=sensor_config['range'],
            rate=sensor_config['rate'],
            x=sensor_config['x'],
            y=sensor_config['y'],
            name=sensor_config.get('name')
        )
        sensors.append(sensor)
    
    # Parse detection model
    detection_config = config.get('detection_model', {})
    detection_model = DetectionModel(
        type=detection_config.get('type', 'gaussian'),
        range_decay=detection_config.get('range_decay', 0.5),
        noise_floor=detection_config.get('noise_floor', 0.01)
    )
    
    # Parse search pattern
    pattern_config = config.get('search_pattern', {})
    search_pattern = SearchPattern(
        name=pattern_config.get('name', 'Default Search Pattern'),
        description=pattern_config.get('description', ''),
        sensors=sensors,
        detection_model=detection_model
    )
    
    return search_pattern


def create_default_search_pattern() -> SearchPattern:
    """Create a default search pattern for testing."""
    sensors = [
        Sensor(time=30.0, range=1000.0, rate=5.0, x=0.0, y=0.0, name="Center"),
        Sensor(time=30.0, range=1500.0, rate=8.0, x=2000.0, y=1000.0, name="North"),
        Sensor(time=60.0, range=1200.0, rate=3.0, x=-1500.0, y=-800.0, name="South")
    ]
    
    detection_model = DetectionModel()
    
    return SearchPattern(
        name="Three-Stage Search",
        description="Progressive sensor deployment with increasing coverage",
        sensors=sensors,
        detection_model=detection_model
    )
