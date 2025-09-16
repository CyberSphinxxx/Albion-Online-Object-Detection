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

@dataclass
class Detection:
    class_id: int
    confidence: float
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    timestamp: float

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
        
        # Configuration
        self.confidence_threshold = 0.6
        self.activity_timeout = 10.0  # seconds before random movement
        self.exploration_duration = 8.0  # increased exploration time for better walking
        self.max_history_size = 100
        self.post_click_wait = 10.0  # 10 second wait after clicking as requested
        
        # Priority system for different object classes
        self.class_priorities = {
            0: 1.0,  # Default priority
            # Add specific class priorities as needed
            # e.g., 1: 2.0 for high priority objects
        }
        
        # Statistics tracking
        self.stats = {
            'total_detections': 0,
            'total_clicks': 0,
            'total_right_clicks': 0,  # track right clicks for movement
            'exploration_sessions': 0,
            'start_time': time.time()
        }
        
        # Load previous session data if exists
        self.load_session_data()
        
        print("🤖 Smart Educational Bot initialized!")
        print("Features enabled:")
        print("  ✓ Smart detection prioritization")
        print("  ✓ Right-click walking movement")  # new feature
        print("  ✓ Random exploration after 10s inactivity")
        print("  ✓ 10-second wait after object interaction")  # new feature
        print("  ✓ Click history tracking")
        print("  ✓ Adaptive confidence scoring")
        print("  ✓ Session persistence")
        print("  ✓ Performance statistics")
        print("\nPress 'q' to quit, 'p' to pause, 's' for stats")

    def load_session_data(self):
        """Load previous session data for continuity."""
        try:
            if os.path.exists('bot_session.json'):
                with open('bot_session.json', 'r') as f:
                    data = json.load(f)
                    self.stats.update(data.get('stats', {}))
                    self.class_priorities.update(data.get('class_priorities', {}))
                print("📊 Previous session data loaded")
        except Exception as e:
            print(f"⚠️ Could not load session data: {e}")

    def save_session_data(self):
        """Save session data for next time."""
        try:
            data = {
                'stats': self.stats,
                'class_priorities': self.class_priorities,
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

    def perform_walking_movement(self, target_x: int = None, target_y: int = None):
        """Perform right-click walking movement to specified location or random location."""
        if target_x is None or target_y is None:
            # Generate random walking destination
            target_x = random.randint(100, self.monitor['width'] - 100)
            target_y = random.randint(100, self.monitor['height'] - 100)
        
        print(f"🚶 Walking to ({target_x}, {target_y})")
        
        # Move mouse to target location smoothly
        self.move_mouse_smoothly(target_x, target_y)
        self.random_delay(0.2, 0.3)
        
        # Right-click to initiate walking
        pyautogui.rightClick()
        self.stats['total_right_clicks'] += 1
        
        # Wait for walking animation/movement to complete
        self.random_delay(1.5, 2.0)
        
        self.last_activity_time = time.time()

    def explore_randomly(self):
        """Perform random exploration with walking movements when no targets are detected."""
        print("🎯 Entering exploration mode with walking...")
        self.exploration_mode = True
        self.stats['exploration_sessions'] += 1
        
        start_time = time.time()
        walk_count = 0
        
        while time.time() - start_time < self.exploration_duration and self.running:
            self.perform_walking_movement()
            walk_count += 1
            
            print(f"🚶 Walk #{walk_count} completed")
            
            self.random_delay(2.0, 3.0)
            
            # Check for objects during exploration
            if walk_count % 2 == 0:  # Every other walk, do a quick scan
                img = np.array(self.sct.grab(self.monitor))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                results = self.model(img_rgb)
                
                # If we find something during exploration, exit early
                for r in results:
                    if len(r.boxes) > 0 and r.boxes.conf.cpu().numpy().max() > self.confidence_threshold:
                        print("🎯 Object detected during exploration, exiting walk mode")
                        self.exploration_mode = False
                        return
        
        self.exploration_mode = False
        self.last_activity_time = time.time()
        print(f"✅ Exploration complete - performed {walk_count} walks")

    def process_detections(self, detections: List[Detection]) -> Optional[Detection]:
        """Process and prioritize detections to select the best target."""
        if not detections:
            return None
        
        # Calculate priority scores for all detections
        scored_detections = [
            (detection, self.calculate_priority_score(detection))
            for detection in detections
        ]
        
        # Sort by priority score (highest first)
        scored_detections.sort(key=lambda x: x[1], reverse=True)
        
        best_detection, best_score = scored_detections[0]
        
        print(f"🎯 Best target: class={best_detection.class_id}, "
              f"confidence={best_detection.confidence:.2f}, "
              f"priority_score={best_score:.2f}")
        
        return best_detection

    def perform_click(self, detection: Detection):
        """Perform a click on the detected object with extended wait time."""
        print(f"🖱️ Clicking on class {detection.class_id} at {detection.center}")
        
        self.move_mouse_smoothly(*detection.center)
        self.random_delay(0.1, 0.2)
        
        # Different click patterns based on object type (educational enhancement)
        if detection.class_id in [0, 1]:  # Example: special handling for certain classes
            pyautogui.click(clicks=2, interval=0.1)  # Double click
        else:
            pyautogui.click()  # Left click as requested
        
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
        
        print(f"⏳ Waiting {self.post_click_wait} seconds after interaction...")
        time.sleep(self.post_click_wait)
        
        self.last_activity_time = time.time()

    def print_statistics(self):
        """Print current bot statistics."""
        uptime = time.time() - self.stats['start_time']
        print("\n" + "="*50)
        print("📊 BOT STATISTICS")
        print("="*50)
        print(f"🕐 Uptime: {uptime:.1f} seconds")
        print(f"👁️ Total detections: {self.stats['total_detections']}")
        print(f"🖱️ Total left clicks: {self.stats['total_clicks']}")
        print(f"🚶 Total right clicks (walks): {self.stats['total_right_clicks']}")  # new stat
        print(f"🎯 Click success rate: {(self.stats['total_clicks']/max(1, self.stats['total_detections'])*100):.1f}%")
        print(f"🚶 Exploration sessions: {self.stats['exploration_sessions']}")
        print(f"📈 Avg interactions/minute: {(self.stats['total_clicks']/(uptime/60)):.1f}")
        print("="*50 + "\n")

    def run(self):
        """Main bot execution loop with improved search-move-interact cycle."""
        try:
            print("🚀 Starting continuous bot loop...")
            
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
                    # INTERACT: Objects detected, process and click
                    best_detection = self.process_detections(current_detections)
                    if best_detection:
                        print("🎯 INTERACT MODE: Object detected, engaging...")
                        self.perform_click(best_detection)
                        # Note: perform_click now includes the 10-second wait
                else:
                    # SEARCH & MOVE: No objects detected, check if we should explore
                    time_since_activity = time.time() - self.last_activity_time
                    if time_since_activity > self.activity_timeout and not self.exploration_mode:
                        print("🔍 SEARCH MODE: No objects found, starting random walk...")
                        self.explore_randomly()
                    elif not self.exploration_mode:
                        if random.random() < 0.1:  # 10% chance for spontaneous movement
                            print("🚶 MOVE MODE: Spontaneous walk...")
                            self.perform_walking_movement()

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("🛑 Quit command received")
                    break
                elif key == ord("p"):
                    print("⏸️ Pausing for 3 seconds...")
                    time.sleep(3)
                    self.last_activity_time = time.time()  # Reset activity timer
                elif key == ord("s"):
                    self.print_statistics()

                # Small delay to prevent excessive CPU usage
                time.sleep(0.05)
                
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
        cv2.destroyAllWindows()
        print("✅ Bot shutdown complete")

def main():
    # Configuration
    MODEL_PATH = "runs/detect/train3/weights/best.pt"  # Adjust to your path
    MONITOR_CONFIG = {
        "top": 0, 
        "left": 0, 
        "width": 1920, 
        "height": 1080
    }  # Adjust for your screen
    
    # Initialize and run bot
    bot = SmartEducationalBot(MODEL_PATH, MONITOR_CONFIG)
    bot.run()

if __name__ == "__main__":
    main()
