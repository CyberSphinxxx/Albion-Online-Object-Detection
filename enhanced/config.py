"""
Configuration file for the Enhanced Gaming Bot
Allows easy customization of bot behavior without modifying main code
"""

class BotConfig:
    # Core Detection Settings
    CONFIDENCE_THRESHOLD = 0.6
    DETECTION_AREA_SIZE = 300  # Size of detection rectangle/cone
    
    # Timing Settings
    POST_CLICK_WAIT = 10.0  # Wait time after clicking objects (seconds)
    ACTIVITY_TIMEOUT = 10.0  # Time before starting exploration (seconds)
    EXPLORATION_DURATION = 8.0  # How long to explore when no objects found
    IDLE_FREQUENCY = 30.0  # Time between idle actions (seconds)
    MEMORY_DECAY_TIME = 300.0  # Time before areas can be revisited (seconds)
    
    # Search Pattern Settings
    SEARCH_PATTERNS = ["spiral", "grid", "random"]
    DEFAULT_SEARCH_PATTERN = "spiral"
    
    # Spiral Pattern Settings
    SPIRAL_ANGLE_INCREMENT = 30  # Degrees to increment spiral angle
    SPIRAL_RADIUS_INCREMENT = 50  # Pixels to expand spiral radius
    SPIRAL_MAX_RADIUS = 300  # Maximum spiral radius before reset
    
    # Grid Pattern Settings
    GRID_SIZE = 150  # Distance between grid points
    GRID_MARGIN = 100  # Margin from screen edges
    
    # Memory System Settings
    MAX_EXPLORED_AREAS = 50  # Maximum number of areas to remember
    AREA_EXPLORATION_RADIUS = 75  # Radius of explored area markers
    
    # Object Priority Settings
    CLASS_PRIORITIES = {
        0: 1.0,  # Default priority
        # Add specific class priorities here
        # Example: 1: 2.0,  # High priority for class 1
        # Example: 2: 0.5,  # Low priority for class 2
    }
    
    # Idle Behavior Settings
    IDLE_ACTIONS = ["scan", "pause", "micro_movement"]
    IDLE_SCAN_RADIUS = 100  # Radius for scanning movements
    IDLE_PAUSE_MIN = 2.0  # Minimum pause time
    IDLE_PAUSE_MAX = 4.0  # Maximum pause time
    IDLE_MOVEMENT_RANGE = 50  # Range for micro movements
    
    # Vision System Settings
    VISION_ENABLED_BY_DEFAULT = True
    DETECTION_BOX_THICKNESS = 2
    PRIORITY_BOX_THICKNESS = 3
    DETECTION_COLOR = (255, 0, 0)  # Blue for regular detections
    PRIORITY_COLOR = (0, 255, 0)  # Green for priority target
    DETECTION_AREA_COLOR = (0, 255, 255)  # Yellow for detection area
    EXPLORED_AREA_COLOR = (100, 100, 100)  # Gray for explored areas
    
    # Screen Settings (adjust for your monitor)
    MONITOR_CONFIG = {
        "top": 0,
        "left": 0, 
        "width": 1920,
        "height": 1080
    }
    
    # Model Settings
    MODEL_PATH = "runs/detect/train3/weights/best.pt"  # Path to your YOLO model
    
    # Statistics and Logging
    MAX_HISTORY_SIZE = 100  # Maximum number of clicks to remember
    SAVE_SESSION_DATA = True  # Whether to save/load session data
    
    @classmethod
    def get_search_pattern_config(cls, pattern_name: str) -> dict:
        """Get configuration for a specific search pattern."""
        configs = {
            "spiral": {
                "angle_increment": cls.SPIRAL_ANGLE_INCREMENT,
                "radius_increment": cls.SPIRAL_RADIUS_INCREMENT,
                "max_radius": cls.SPIRAL_MAX_RADIUS
            },
            "grid": {
                "grid_size": cls.GRID_SIZE,
                "margin": cls.GRID_MARGIN
            },
            "random": {
                "avoid_recent": True,
                "exploration_radius": cls.AREA_EXPLORATION_RADIUS
            }
        }
        return configs.get(pattern_name, {})
