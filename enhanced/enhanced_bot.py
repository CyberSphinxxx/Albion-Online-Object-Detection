import pyautogui
import time
import random
import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import threading
import queue
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
import os
import math

@dataclass
class Detection:
    class_id: int
    confidence: float
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    timestamp: float
    distance_from_center: float = 0.0

@dataclass
class ExploredArea:
    center: Tuple[int, int]
    radius: int
    timestamp: float
    visit_count: int = 1

class SmartEducationalBot:
    def __init__(self, model_path: str, monitor_config: dict):
        # Load YOLO model
        self.model = YOLO(model_path)
        self.monitor = monitor_config
        self.sct = mss()
        
        # Bot state management
        self.last_activity_time = time.time()
        self.detection_history = []
        self.click_history = []
        self.running = True
        self.exploration_mode = False
        self.idle_mode = False
        
        self.vision_window_active = True  # Enable by default so GUI loads
        self.vision_window_created = False
        self.vision_update_interval = 0.2  # Update vision every 200ms instead of every frame
        self.last_vision_update = 0
        
        self.explored_areas = []  # Memory system for explored areas
        self.recent_clicks = []  # Track recent clicks with better cooldown
        self.click_cooldown_radius = 150  # Increased radius to avoid same area
        self.click_cooldown_time = 30.0  # 30 seconds before clicking same area again
        self.current_search_pattern = "spiral"  # spiral, grid, random
        self.search_center = None
        self.spiral_angle = 0
        self.spiral_radius = 50
        self.grid_positions = []
        self.current_grid_index = 0
        
        # Configuration
        self.confidence_threshold = 0.6
        self.activity_timeout = 10.0  # seconds before random movement
        self.exploration_duration = 8.0
        self.max_history_size = 100
        self.post_click_wait = 10.0  # 10 second wait after clicking
        
        self.memory_decay_time = 300.0  # 5 minutes before areas can be revisited
        self.detection_area_size = 300  # Size of detection cone/rectangle
        
        self.search_patterns = ["spiral", "grid", "random"]
        self.idle_actions = ["scan", "pause", "micro_movement"]
        self.last_idle_time = time.time()
        self.idle_frequency = 30.0  # seconds between idle actions
        
        # Priority system for different object classes
        self.class_priorities = {
            0: 1.0,  # Default priority
        }
        
        # Statistics tracking
        self.stats = {
            'total_detections': 0,
            'total_clicks': 0,
            'total_right_clicks': 0,
            'total_mounts': 0,  # Track "A" key presses for mounting
            'exploration_sessions': 0,
            'idle_actions': 0,
            'areas_explored': 0,
            'start_time': time.time()
        }
        
        # Load previous session data if exists
        self.load_session_data()
        
        print("🤖 Enhanced Smart Educational Bot initialized!")
        print("Features enabled:")
        print("  ✓ Smart detection prioritization")
        print("  ✓ Right-click walking movement")
        print("  ✓ Vision system with detection area")  # new
        print("  ✓ Memory system for explored areas")  # new
        print("  ✓ Advanced search patterns (spiral/grid/random)")  # new
        print("  ✓ Automatic mounting after interactions")  # new
        print("  ✓ Idle behavior simulation")  # new
        print("  ✓ 10-second wait after object interaction")
        print("  ✓ Click history tracking")
        print("  ✓ Adaptive confidence scoring")
        print("  ✓ Session persistence")
        print("  ✓ Performance statistics")
        print("\nPress 'q' to quit, 'p' to pause, 's' for stats, 'v' to toggle vision")

    def load_session_data(self):
        """Load previous session data for continuity."""
        try:
            if os.path.exists('bot_session.json'):
                with open('bot_session.json', 'r') as f:
                    data = json.load(f)
                    self.stats.update(data.get('stats', {}))
                    self.class_priorities.update(data.get('class_priorities', {}))
                    saved_areas = data.get('explored_areas', [])
                    current_time = time.time()
                    for area_data in saved_areas:
                        if current_time - area_data['timestamp'] < self.memory_decay_time:
                            self.explored_areas.append(ExploredArea(
                                center=tuple(area_data['center']),
                                radius=area_data['radius'],
                                timestamp=area_data['timestamp'],
                                visit_count=area_data.get('visit_count', 1)
                            ))
                print("📊 Previous session data loaded")
        except Exception as e:
            print(f"⚠️ Could not load session data: {e}")

    def save_session_data(self):
        """Save session data for next time."""
        try:
            areas_data = []
            for area in self.explored_areas:
                areas_data.append({
                    'center': area.center,
                    'radius': area.radius,
                    'timestamp': area.timestamp,
                    'visit_count': area.visit_count
                })
            
            data = {
                'stats': self.stats,
                'class_priorities': self.class_priorities,
                'explored_areas': areas_data,
                'last_session': time.time()
            }
            with open('bot_session.json', 'w') as f:
                json.dump(data, f, indent=2)
            print("💾 Session data saved")
        except Exception as e:
            print(f"⚠️ Could not save session data: {e}")

    def random_delay(self, base: float = 0.2, jitter: float = 0.3):
        """Add human-like random delays."""
        time.sleep(base + random.uniform(0, jitter))

    def move_mouse_smoothly(self, x: int, y: int):
        """Move mouse with natural human-like movement."""
        # Add slight randomness to avoid exact positioning
        target_x = x + random.randint(-5, 5)
        target_y = y + random.randint(-5, 5)
        
        # Ensure coordinates are within screen bounds
        target_x = max(0, min(self.monitor['width'] - 1, target_x))
        target_y = max(0, min(self.monitor['height'] - 1, target_y))
        
        # Variable duration based on distance
        current_x, current_y = pyautogui.position()
        distance = ((target_x - current_x) ** 2 + (target_y - current_y) ** 2) ** 0.5
        duration = min(0.8, max(0.1, distance / 1000))
        
        pyautogui.moveTo(target_x, target_y, duration=duration)

    def calculate_priority_score(self, detection: Detection) -> float:
        """Calculate priority score for a detection based on multiple factors."""
        base_score = detection.confidence
        
        # Class-based priority multiplier
        class_priority = self.class_priorities.get(detection.class_id, 1.0)
        base_score *= class_priority
        
        # Avoid clicking the same area repeatedly
        recent_clicks = [click for click in self.click_history[-10:] 
                        if time.time() - click['timestamp'] < 5.0]
        
        for click in recent_clicks:
            distance = ((detection.center[0] - click['position'][0]) ** 2 + 
                       (detection.center[1] - click['position'][1]) ** 2) ** 0.5
            if distance < 100:  # Within 100 pixels
                base_score *= 0.5  # Reduce priority
        
        # Prefer objects closer to center (optional behavioral bias)
        center_x, center_y = self.monitor['width'] // 2, self.monitor['height'] // 2
        distance_from_center = ((detection.center[0] - center_x) ** 2 + 
                               (detection.center[1] - center_y) ** 2) ** 0.5
        max_distance = ((center_x ** 2) + (center_y ** 2)) ** 0.5
        center_bias = 1.0 - (distance_from_center / max_distance) * 0.2
        base_score *= center_bias
        
        return base_score

    def is_area_recently_explored(self, x: int, y: int, radius: int = 100) -> bool:
        """Check if an area has been recently explored."""
        current_time = time.time()
        for area in self.explored_areas:
            if current_time - area.timestamp > self.memory_decay_time:
                continue  # Area memory has decayed
            
            distance = math.sqrt((x - area.center[0])**2 + (y - area.center[1])**2)
            if distance < (radius + area.radius):
                return True
        return False

    def mark_area_explored(self, x: int, y: int, radius: int = 100):
        """Mark an area as explored in memory."""
        # Check if we're updating an existing area
        for area in self.explored_areas:
            distance = math.sqrt((x - area.center[0])**2 + (y - area.center[1])**2)
            if distance < radius:
                area.timestamp = time.time()
                area.visit_count += 1
                return
        
        # Add new explored area
        self.explored_areas.append(ExploredArea(
            center=(x, y),
            radius=radius,
            timestamp=time.time()
        ))
        self.stats['areas_explored'] += 1
        
        # Limit memory size
        if len(self.explored_areas) > 50:
            self.explored_areas = sorted(self.explored_areas, key=lambda a: a.timestamp)[-40:]

    def get_detection_area_bounds(self) -> Tuple[int, int, int, int]:
        """Get the bounds of the detection area (rectangular for now)."""
        center_x = self.monitor['width'] // 2
        center_y = self.monitor['height'] // 2
        half_size = self.detection_area_size // 2
        
        return (
            max(0, center_x - half_size),
            max(0, center_y - half_size),
            min(self.monitor['width'], center_x + half_size),
            min(self.monitor['height'], center_y + half_size)
        )

    def is_in_detection_area(self, x: int, y: int) -> bool:
        """Check if coordinates are within the detection area."""
        x1, y1, x2, y2 = self.get_detection_area_bounds()
        return x1 <= x <= x2 and y1 <= y <= y2

    def create_vision_display(self, img: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Create a vision display showing what the bot sees."""
        try:
            height, width = img.shape[:2]
            scale_factor = 0.6  # Scale down to 60% of original size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            vision_img = cv2.resize(img, (new_width, new_height))
            
            # Adjust coordinates for scaled image
            scale_x = new_width / width
            scale_y = new_height / height
            
            current_time = time.time()
            for click in self.recent_clicks:
                if current_time - click['timestamp'] < self.click_cooldown_time:
                    scaled_center = (int(click['x'] * scale_x), int(click['y'] * scale_y))
                    scaled_radius = int(self.click_cooldown_radius * scale_x)
                    alpha = 1.0 - (current_time - click['timestamp']) / self.click_cooldown_time
                    color = (0, 0, 255)  # Red for cooldown areas
                    cv2.circle(vision_img, scaled_center, scaled_radius, color, 1)
                    # Add cooldown timer text
                    remaining = int(self.click_cooldown_time - (current_time - click['timestamp']))
                    cv2.putText(vision_img, f"{remaining}s", scaled_center, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw detection area (scaled)
            x1, y1, x2, y2 = self.get_detection_area_bounds()
            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
            cv2.rectangle(vision_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(vision_img, "DETECTION AREA", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Draw explored areas (scaled and simplified)
            current_time = time.time()
            for area in self.explored_areas[-10:]:  # Only show last 10 areas to reduce clutter
                if current_time - area.timestamp < self.memory_decay_time:
                    scaled_center = (int(area.center[0] * scale_x), int(area.center[1] * scale_y))
                    scaled_radius = int(area.radius * scale_x)
                    alpha = max(0.1, 1.0 - (current_time - area.timestamp) / self.memory_decay_time)
                    color = (100, 100, 100)
                    cv2.circle(vision_img, scaled_center, scaled_radius, color, 1)
            
            # Highlight detections (scaled)
            closest_detection = None
            min_distance = float('inf')
            
            for detection in detections:
                # Scale detection coordinates
                scaled_center = (int(detection.center[0] * scale_x), int(detection.center[1] * scale_y))
                scaled_bbox = (
                    int(detection.bbox[0] * scale_x), int(detection.bbox[1] * scale_y),
                    int(detection.bbox[2] * scale_x), int(detection.bbox[3] * scale_y)
                )
                
                # Calculate distance from center for prioritization
                center_x, center_y = self.monitor['width'] // 2, self.monitor['height'] // 2
                distance = math.sqrt((detection.center[0] - center_x)**2 + (detection.center[1] - center_y)**2)
                detection.distance_from_center = distance
                
                if distance < min_distance:
                    min_distance = distance
                    closest_detection = detection
                
                # Draw detection box (scaled)
                x1, y1, x2, y2 = scaled_bbox
                color = (0, 255, 0) if detection == closest_detection else (255, 0, 0)
                thickness = 2 if detection == closest_detection else 1
                
                cv2.rectangle(vision_img, (x1, y1), (x2, y2), color, thickness)
                cv2.circle(vision_img, scaled_center, 3, color, -1)
                
                # Add simplified labels
                if detection == closest_detection:
                    label = f"TARGET: {detection.confidence:.2f}"
                    cv2.putText(vision_img, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Add simplified status information
            status_y = 20
            cv2.putText(vision_img, f"Mode: {'EXPLORE' if self.exploration_mode else 'IDLE' if self.idle_mode else 'HUNT'}", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vision_img, f"Pattern: {self.current_search_pattern.upper()}", 
                       (10, status_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vision_img, f"Cooldowns: {len(self.recent_clicks)}", 
                       (10, status_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return vision_img
        except Exception as e:
            print(f"⚠️ Vision display error: {e}")
            # Return a simple black image if there's an error
            return np.zeros((400, 600, 3), dtype=np.uint8)

    def perform_mount_action(self):
        """Press 'A' to mount up after interaction."""
        print("🐎 Mounting up (pressing A)...")
        pyautogui.press('a')
        self.stats['total_mounts'] += 1
        self.random_delay(1.0, 1.5)  # Wait for mount animation

    def perform_idle_action(self):
        """Perform an idle action to simulate natural behavior."""
        if time.time() - self.last_idle_time < self.idle_frequency:
            return
        
        self.idle_mode = True
        action = random.choice(self.idle_actions)
        
        if action == "scan":
            print("👁️ Idle, scanning area...")
            # Look around by moving mouse in a circle
            center_x, center_y = self.monitor['width'] // 2, self.monitor['height'] // 2
            for angle in range(0, 360, 45):
                x = center_x + int(100 * math.cos(math.radians(angle)))
                y = center_y + int(100 * math.sin(math.radians(angle)))
                pyautogui.moveTo(x, y, duration=0.3)
                time.sleep(0.2)
        
        elif action == "pause":
            print("⏸️ Idle, pausing to observe...")
            time.sleep(random.uniform(2.0, 4.0))
        
        elif action == "micro_movement":
            print("🔄 Idle, small adjustment movement...")
            current_x, current_y = pyautogui.position()
            new_x = current_x + random.randint(-50, 50)
            new_y = current_y + random.randint(-50, 50)
            self.perform_walking_movement(new_x, new_y)
        
        self.stats['idle_actions'] += 1
        self.last_idle_time = time.time()
        self.idle_mode = False

    def get_next_search_position(self) -> Tuple[int, int]:
        """Get next position based on current search pattern."""
        if self.search_center is None:
            self.search_center = (self.monitor['width'] // 2, self.monitor['height'] // 2)
        
        if self.current_search_pattern == "spiral":
            # Spiral pattern
            x = self.search_center[0] + int(self.spiral_radius * math.cos(math.radians(self.spiral_angle)))
            y = self.search_center[1] + int(self.spiral_radius * math.sin(math.radians(self.spiral_angle)))
            
            self.spiral_angle += 30  # Increase angle
            if self.spiral_angle >= 360:
                self.spiral_angle = 0
                self.spiral_radius += 50  # Expand spiral
                if self.spiral_radius > 300:  # Reset spiral
                    self.spiral_radius = 50
            
            return self.clamp_to_screen(x, y)
        
        elif self.current_search_pattern == "grid":
            # Grid pattern
            if not self.grid_positions:
                # Generate grid positions
                grid_size = 150
                for x in range(100, self.monitor['width'] - 100, grid_size):
                    for y in range(100, self.monitor['height'] - 100, grid_size):
                        if not self.is_area_recently_explored(x, y, 75):
                            self.grid_positions.append((x, y))
                random.shuffle(self.grid_positions)
            
            if self.grid_positions:
                pos = self.grid_positions.pop(0)
                return pos
            else:
                # Regenerate grid if empty
                self.grid_positions = []
                return self.get_next_search_position()
        
        else:  # random
            # Random pattern avoiding recently explored areas
            attempts = 0
            while attempts < 10:
                x = random.randint(100, self.monitor['width'] - 100)
                y = random.randint(100, self.monitor['height'] - 100)
                if not self.is_area_recently_explored(x, y, 100):
                    return (x, y)
                attempts += 1
            
            # If all areas recently explored, pick random anyway
            return (random.randint(100, self.monitor['width'] - 100),
                   random.randint(100, self.monitor['height'] - 100))

    def clamp_to_screen(self, x: int, y: int) -> Tuple[int, int]:
        """Ensure coordinates are within screen bounds with margin."""
        margin = 50
        x = max(margin, min(self.monitor['width'] - margin, x))
        y = max(margin, min(self.monitor['height'] - margin, y))
        return (x, y)

    def perform_walking_movement(self, target_x: int = None, target_y: int = None):
        """Perform right-click walking movement to specified location or search pattern location."""
        if target_x is None or target_y is None:
            attempts = 0
            while attempts < 10:
                target_x, target_y = self.get_next_search_position()
                if not self.is_recently_clicked(target_x, target_y):
                    break
                attempts += 1
        
        target_x, target_y = self.clamp_to_screen(target_x, target_y)
        
        print(f"🚶 Walking to ({target_x}, {target_y}) using {self.current_search_pattern} pattern")
        
        # Move mouse to target location smoothly
        self.move_mouse_smoothly(target_x, target_y)
        self.random_delay(0.2, 0.3)
        
        # Right-click to initiate walking
        pyautogui.rightClick()
        self.stats['total_right_clicks'] += 1
        
        # Mark area as explored
        self.mark_area_explored(target_x, target_y, 75)
        
        # Wait for walking animation/movement to complete
        self.random_delay(1.5, 2.0)
        
        self.last_activity_time = time.time()

    def explore_randomly(self):
        """Perform structured exploration using current search pattern."""
        print(f"🎯 Entering exploration mode using {self.current_search_pattern} pattern...")
        self.exploration_mode = True
        self.stats['exploration_sessions'] += 1
        
        start_time = time.time()
        walk_count = 0
        
        while time.time() - start_time < self.exploration_duration and self.running:
            self.perform_walking_movement()
            walk_count += 1
            
            print(f"🚶 Walk #{walk_count} completed ({self.current_search_pattern} pattern)")
            
            self.random_delay(2.0, 3.0)
            
            # Check for objects during exploration
            if walk_count % 2 == 0:  # Every other walk, do a quick scan
                img = np.array(self.sct.grab(self.monitor))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                results = self.model(img_rgb)
                
                # If we find something during exploration, exit early
                for r in results:
                    if len(r.boxes) > 0 and r.boxes.conf.cpu().numpy().max() > self.confidence_threshold:
                        print("🎯 Object detected during exploration, exiting search mode")
                        self.exploration_mode = False
                        return
        
        # Occasionally change search pattern
        if random.random() < 0.3:  # 30% chance to change pattern
            old_pattern = self.current_search_pattern
            self.current_search_pattern = random.choice(self.search_patterns)
            if old_pattern != self.current_search_pattern:
                print(f"🔄 Switching search pattern: {old_pattern} → {self.current_search_pattern}")
                # Reset pattern-specific variables
                self.spiral_angle = 0
                self.spiral_radius = 50
                self.grid_positions = []
        
        self.exploration_mode = False
        self.last_activity_time = time.time()
        print(f"✅ Exploration complete - performed {walk_count} walks using {self.current_search_pattern}")

    def process_detections(self, detections: List[Detection]) -> Optional[Detection]:
        """Process and prioritize detections to select the best target (closest first)."""
        if not detections:
            return None
        
        # Filter out recently clicked areas
        valid_detections = []
        for detection in detections:
            if not self.is_recently_clicked(detection.center[0], detection.center[1]):
                valid_detections.append(detection)
        
        if not valid_detections:
            print("🚫 All detected objects are in recently clicked areas, exploring...")
            return None
        
        # Calculate distances and priority scores
        center_x, center_y = self.monitor['width'] // 2, self.monitor['height'] // 2
        
        for detection in valid_detections:
            distance = math.sqrt((detection.center[0] - center_x)**2 + (detection.center[1] - center_y)**2)
            detection.distance_from_center = distance
        
        # Sort by distance (closest first) then by confidence
        valid_detections.sort(key=lambda d: (d.distance_from_center, -d.confidence))
        
        best_detection = valid_detections[0]
        
        print(f"🎯 Object detected → approaching...")
        print(f"   Target: class={best_detection.class_id}, confidence={best_detection.confidence:.2f}")
        print(f"   Distance from center: {best_detection.distance_from_center:.1f}px")
        
        return best_detection

    def perform_click(self, detection: Detection):
        """Perform a click on the detected object with extended wait time and mounting."""
        # Record this click location to prevent repetitive clicking
        self.record_click(detection.center[0], detection.center[1])
        
        print(f"🖱️ Clicking object at ({detection.center[0]}, {detection.center[1]})...")
        
        self.move_mouse_smoothly(*detection.center)
        self.random_delay(0.3, 0.5)  # Slightly longer delay before clicking
        
        # Left click for mining/interaction
        print(f"🖱️ Performing left click...")
        pyautogui.click()
        
        # Log the click
        self.click_history.append({
            'timestamp': time.time(),
            'position': detection.center,
            'class_id': detection.class_id,
            'confidence': detection.confidence
        })
        
        # Maintain history size
        if len(self.click_history) > self.max_history_size:
            self.click_history = self.click_history[-self.max_history_size:]
        
        self.stats['total_clicks'] += 1
        
        print(f"🖱️ Mining... waiting {self.post_click_wait}s")
        
        # Wait the specified time after clicking
        time.sleep(self.post_click_wait)
        
        self.perform_mount_action()
        
        self.last_activity_time = time.time()

    def print_statistics(self):
        """Print current bot statistics."""
        uptime = time.time() - self.stats['start_time']
        print("\n" + "="*60)
        print("📊 ENHANCED BOT STATISTICS")
        print("="*60)
        print(f"🕐 Uptime: {uptime:.1f} seconds")
        print(f"👁️ Total detections: {self.stats['total_detections']}")
        print(f"🖱️ Total left clicks: {self.stats['total_clicks']}")
        print(f"🚶 Total right clicks (walks): {self.stats['total_right_clicks']}")
        print(f"🐎 Total mounts: {self.stats['total_mounts']}")  # new stat
        print(f"🎯 Click success rate: {(self.stats['total_clicks']/max(1, self.stats['total_detections'])*100):.1f}%")
        print(f"🚶 Exploration sessions: {self.stats['exploration_sessions']}")
        print(f"🗺️ Areas explored: {self.stats['areas_explored']}")  # new stat
        print(f"😴 Idle actions performed: {self.stats['idle_actions']}")  # new stat
        print(f"🔍 Current search pattern: {self.current_search_pattern}")  # new stat
        print(f"📈 Avg interactions/minute: {(self.stats['total_clicks']/(uptime/60)):.1f}")
        print("="*60 + "\n")

    def run(self):
        """Main bot execution loop with enhanced vision system and behaviors."""
        try:
            print("🚀 Starting enhanced bot loop...")
            print("👁️ Vision system enabled by default - press 'v' to toggle")
            
            while self.running:
                # Capture screen
                img = np.array(self.sct.grab(self.monitor))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # Run YOLO detection
                results = self.model(img_rgb)
                
                current_detections = []
                
                # Process YOLO results
                for r in results:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confidences = r.boxes.conf.cpu().numpy()
                    class_ids = r.boxes.cls.cpu().numpy()

                    for i, box in enumerate(boxes):
                        conf = confidences[i]
                        class_id = int(class_ids[i])

                        if conf < self.confidence_threshold:
                            continue

                        x1, y1, x2, y2 = box
                        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        
                        detection = Detection(
                            class_id=class_id,
                            confidence=conf,
                            center=center,
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            timestamp=time.time()
                        )
                        
                        current_detections.append(detection)
                        self.stats['total_detections'] += 1

                if current_detections:
                    print(f"🔍 Found {len(current_detections)} objects with confidence > {self.confidence_threshold}")

                current_time = time.time()
                if self.vision_window_active and (current_time - self.last_vision_update) > self.vision_update_interval:
                    try:
                        vision_img = self.create_vision_display(img_rgb, current_detections)
                        
                        # Create window only once and position it safely
                        if not self.vision_window_created:
                            cv2.namedWindow("Bot Vision", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                            cv2.resizeWindow("Bot Vision", 800, 600)
                            try:
                                cv2.moveWindow("Bot Vision", 100, 100)  # Position window
                            except:
                                pass  # Ignore if window positioning fails
                            self.vision_window_created = True
                            print("👁️ Vision window created successfully")
                        
                        cv2.imshow("Bot Vision", vision_img)
                        self.last_vision_update = current_time
                    except Exception as e:
                        print(f"⚠️ Vision system error: {e}")
                        # Don't disable vision system on error, just skip this frame
                        pass

                # Check for objects during exploration
                if current_detections:
                    # INTERACT: Objects detected, process and click
                    best_detection = self.process_detections(current_detections)
                    if best_detection:
                        self.perform_click(best_detection)
                        # Note: perform_click now includes mounting
                else:
                    # SEARCH & MOVE: No objects detected, check behaviors
                    time_since_activity = time.time() - self.last_activity_time
                    
                    # Perform idle actions occasionally
                    self.perform_idle_action()
                    
                    if time_since_activity > self.activity_timeout and not self.exploration_mode:
                        print("🔍 No objects found, starting structured search...")
                        self.explore_randomly()
                    elif not self.exploration_mode and random.random() < 0.05:  # 5% chance
                        print("🚶 Spontaneous movement...")
                        self.perform_walking_movement()

                try:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("🛑 Quit command received")
                        break
                    elif key == ord("p"):
                        print("⏸️ Pausing for 3 seconds...")
                        time.sleep(3)
                        self.last_activity_time = time.time()
                    elif key == ord("s"):
                        self.print_statistics()
                    elif key == ord("v"):  # Toggle vision system
                        self.vision_window_active = not self.vision_window_active
                        if not self.vision_window_active:
                            try:
                                cv2.destroyWindow("Bot Vision")
                            except:
                                pass
                            self.vision_window_created = False
                            print("👁️ Vision system disabled")
                        else:
                            print("👁️ Vision system enabled")
                except:
                    pass  # Ignore keyboard input errors

                time.sleep(0.1)  # Increased from 0.05 to 0.1
                
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
        except Exception as e:
            print(f"❌ Error occurred: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources and save data."""
        print("🧹 Cleaning up...")
        self.running = False
        self.save_session_data()
        self.print_statistics()
        try:
            cv2.destroyAllWindows()
            time.sleep(0.5)  # Give time for windows to close
        except:
            pass  # Ignore cleanup errors
        print("✅ Enhanced bot shutdown complete")

    def is_recently_clicked(self, x: int, y: int) -> bool:
        """Check if an area has been recently clicked to avoid repetitive behavior."""
        current_time = time.time()
        
        # Clean up old clicks first
        self.recent_clicks = [click for click in self.recent_clicks 
                             if current_time - click['timestamp'] < self.click_cooldown_time]
        
        # Check if current position is too close to recent clicks
        for click in self.recent_clicks:
            distance = math.sqrt((x - click['x'])**2 + (y - click['y'])**2)
            if distance < self.click_cooldown_radius:
                time_since = current_time - click['timestamp']
                print(f"🚫 Area recently clicked {time_since:.1f}s ago, skipping...")
                return True
        
        return False

    def record_click(self, x: int, y: int):
        """Record a click location to prevent repetitive clicking."""
        self.recent_clicks.append({
            'x': x,
            'y': y,
            'timestamp': time.time()
        })
        
        # Limit memory size
        if len(self.recent_clicks) > 20:
            self.recent_clicks = self.recent_clicks[-15:]

def main():
    # Configuration
    MODEL_PATH = "runs/detect/train3/weights/best.pt"  # Adjust to your path
    MONITOR_CONFIG = {
        "top": 0, 
        "left": 0, 
        "width": 1920, 
        "height": 1080
    }  # Adjust for your screen
    
    # Initialize and run enhanced bot
    bot = SmartEducationalBot(MODEL_PATH, MONITOR_CONFIG)
    bot.run()

if __name__ == "__main__":
    main()
