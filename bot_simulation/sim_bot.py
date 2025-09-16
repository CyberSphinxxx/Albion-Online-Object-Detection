"""
sim_bot.py

Local 2D simulation environment for safe, offline testing of detection + human-like agent behavior.

Requirements:
    pip install pygame numpy

Run:
    python sim_bot.py

This simulation is ONLY for local testing and education. It does NOT control your OS mouse or any external programs.
"""

import pygame
import random
import math
import time
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

# --- Config ---
SCREEN_W, SCREEN_H = 1280, 720
FPS = 60

RESOURCE_COUNT = 20
RESOURCE_RADIUS = 18
AGENT_RADIUS = 12

# Reaction and humanization parameters (tuneable)
REACTION_TIME_MEAN = 0.26    # seconds before agent reacts to a new detection
REACTION_TIME_STD = 0.08
MOVE_SPEED_MEAN = 260.0      # pixels per second (peak)
MOVE_SPEED_STD = 40.0
ARRIVAL_THRESHOLD = 10       # pixels
CLICK_DELAY_MEAN = 0.12
CLICK_DELAY_STD = 0.05

# Path variability
WAYPOINT_JITTER = 0.15       # fraction of distance for waypoint offset
AIM_JITTER_PIXELS = 3.5      # small aim jitter while approaching
SPLINE_STEPS = 3             # number of intermediate waypoints for path smoothing

LOG_ACTIONS = True
LOG_FILE = "sim_actions_log.jsonl"


# --- Utility functions ---
def rand_normal(mean, std):
    return max(0.0, random.gauss(mean, std))


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp(x, a, b):
    return max(a, min(b, x))


def unit_vector(a, b):
    d = dist(a, b)
    if d == 0:
        return 0.0, 0.0
    return (b[0] - a[0]) / d, (b[1] - a[1]) / d


# --- Data classes ---
@dataclass
class Resource:
    id: int
    pos: Tuple[int, int]
    radius: int = RESOURCE_RADIUS
    picked: bool = False


@dataclass
class Detection:
    class_name: str
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2
    confidence: float


# --- Simulated detector (placeholder) ---
class SimDetector:
    """
    Returns detections from ground truth resources in the simulation.
    Replace with your offline model: run model on screenshot and return similar Detection objects.
    """
    def __init__(self, resources: List[Resource]):
        self.resources = resources

    def detect(self, frame=None) -> List[Detection]:
        detections = []
        for r in self.resources:
            if not r.picked:
                x, y = r.pos
                x1, y1 = x - r.radius, y - r.radius
                x2, y2 = x + r.radius, y + r.radius
                detections.append(Detection("resource", (x1, y1, x2, y2), confidence=0.8 + random.random() * 0.2))
        # random occasional false positives/negatives can be added for realism
        return detections


# --- Agent with human-like movement and clicking ---
class HumanLikeAgent:
    def __init__(self, x, y):
        self.pos = [float(x), float(y)]
        self.target = None
        self.waypoints: List[Tuple[float, float]] = []
        self.state = "idle"
        self.speed = rand_normal(MOVE_SPEED_MEAN, MOVE_SPEED_STD)
        self.last_detection_time = 0.0
        self.next_reaction_delay = rand_normal(REACTION_TIME_MEAN, REACTION_TIME_STD)
        self.click_scheduled_at = None
        self.log_fp = open(LOG_FILE, "a") if LOG_ACTIONS else None

    def schedule_reaction(self):
        self.next_reaction_delay = rand_normal(REACTION_TIME_MEAN, REACTION_TIME_STD)
        self.last_detection_time = time.time()

    def set_target_from_detection(self, det: Detection):
        x1, y1, x2, y2 = det.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        self.target = (cx, cy)
        self.create_waypoints(self.target)
        self.state = "reacting"
        self.schedule_reaction()
        if LOG_ACTIONS:
            self._log({"event": "target_selected", "target": self.target, "t": time.time()})

    def create_waypoints(self, target):
        # create a few intermediate waypoints with jittered offsets for natural paths
        sx, sy = self.pos
        tx, ty = target
        self.waypoints = []
        total_steps = SPLINE_STEPS
        for i in range(1, total_steps + 1):
            alpha = i / (total_steps + 1)
            bx = sx + (tx - sx) * alpha
            by = sy + (ty - sy) * alpha
            # jitter perpendicular to the line
            dx, dy = (tx - sx), (ty - sy)
            perp = (-dy, dx)
            plen = math.hypot(perp[0], perp[1]) or 1.0
            perp_unit = (perp[0] / plen, perp[1] / plen)
            jitter_amount = WAYPOINT_JITTER * dist((sx, sy), (tx, ty))
            jitter = random.uniform(-jitter_amount, jitter_amount)
            bx += perp_unit[0] * jitter
            by += perp_unit[1] * jitter
            self.waypoints.append((bx, by))
        # final target appended
        self.waypoints.append((tx, ty))

    def update(self, dt):
        now = time.time()

        # If reacting, check reaction delay
        if self.state == "reacting":
            if (now - self.last_detection_time) >= self.next_reaction_delay:
                self.state = "moving"
                if LOG_ACTIONS:
                    self._log({"event": "reaction_delay_passed", "delay": self.next_reaction_delay, "t": now})

        # Movement logic
        if self.state == "moving" and self.waypoints:
            wx, wy = self.waypoints[0]
            # add aiming jitter while approaching
            jitter_x = random.uniform(-AIM_JITTER_PIXELS, AIM_JITTER_PIXELS)
            jitter_y = random.uniform(-AIM_JITTER_PIXELS, AIM_JITTER_PIXELS)
            target_x = wx + jitter_x
            target_y = wy + jitter_y

            # dynamic speed profile: accelerate then decelerate near end
            remaining = dist(self.pos, (target_x, target_y))
            peak_speed = self.speed
            # slow down if near final waypoint
            if len(self.waypoints) == 1:
                peak_speed = peak_speed * clamp(remaining / 120.0, 0.25, 1.0)

            ux, uy = unit_vector(self.pos, (target_x, target_y))
            move_dist = clamp(peak_speed * dt, 0, remaining)
            self.pos[0] += ux * move_dist
            self.pos[1] += uy * move_dist

            # reached waypoint?
            if remaining <= ARRIVAL_THRESHOLD:
                self.waypoints.pop(0)
                if LOG_ACTIONS:
                    self._log({"event": "waypoint_reached", "pos": (self.pos[0], self.pos[1]), "t": time.time()})

            # if arrived at final waypoint, schedule click
            if not self.waypoints:
                self.state = "click_wait"
                click_delay = rand_normal(CLICK_DELAY_MEAN, CLICK_DELAY_STD)
                self.click_scheduled_at = time.time() + click_delay
                if LOG_ACTIONS:
                    self._log({"event": "arrived_target", "pos": (self.pos[0], self.pos[1]), "click_in": click_delay, "t": time.time()})

        # Click execution
        if self.state == "click_wait" and self.click_scheduled_at is not None:
            if time.time() >= self.click_scheduled_at:
                # perform simulated click
                self._simulate_click()
                self.click_scheduled_at = None
                self.state = "idle"
                self.target = None
                if LOG_ACTIONS:
                    self._log({"event": "click_executed", "pos": (self.pos[0], self.pos[1]), "t": time.time()})

    def _simulate_click(self):
        # In a real bot you'd issue input here; in this simulator we return the click coords
        return (int(self.pos[0]), int(self.pos[1]))

    def _log(self, obj):
        if not self.log_fp:
            return
        self.log_fp.write(json.dumps(obj) + "\n")
        self.log_fp.flush()

    def close(self):
        if self.log_fp:
            self.log_fp.close()


# --- Main simulation ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 14)

    # create resources
    resources = []
    margin = 40
    for i in range(RESOURCE_COUNT):
        x = random.randint(margin, SCREEN_W - margin)
        y = random.randint(margin, SCREEN_H - margin)
        resources.append(Resource(i, (x, y)))

    detector = SimDetector(resources)
    agent = HumanLikeAgent(SCREEN_W // 2, SCREEN_H // 2)

    running = True
    last_detect_time = 0.0
    DETECT_INTERVAL = 0.18  # seconds between detection runs (simulate model latency)

    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # manual pick via mouse (for debugging)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                # mark close resources as picked
                for r in resources:
                    if not r.picked and dist((mx, my), r.pos) <= r.radius + 4:
                        r.picked = True

        # Every DETECT_INTERVAL run detection and set agent target if idle
        now = time.time()
        if now - last_detect_time >= DETECT_INTERVAL:
            last_detect_time = now
            detections = detector.detect()
            # choose highest-confidence detection that's not picked
            detections = [d for d in detections if d.confidence > 0.3]
            # pick nearest detection to agent
            best = None
            best_d = 1e9
            for d in detections:
                x1, y1, x2, y2 = d.bbox
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                # check corresponding resource picked state
                # (detector uses ground truth so we can find resource)
                # find resource at that loc
                if any(r.picked for r in resources if dist((cx, cy), r.pos) < 8):
                    continue
                ddist = dist((agent.pos[0], agent.pos[1]), (cx, cy))
                if ddist < best_d:
                    best_d = ddist
                    best = d
            if best and agent.state in ("idle",):
                agent.set_target_from_detection(best)

        # update agent
        agent.update(dt)

        # if agent clicked, mark resource as picked (check proximity to any resource)
        if agent.state == "idle" and agent.target is None:
            # check last logged click by reading last line of log (simpler to check distance)
            # Instead, we'll mark resources within ARRIVAL_THRESHOLD as picked
            for r in resources:
                if not r.picked and dist((agent.pos[0], agent.pos[1]), r.pos) < ARRIVAL_THRESHOLD + r.radius:
                    r.picked = True
                    if LOG_ACTIONS:
                        agent._log({"event": "resource_picked", "resource_id": r.id, "t": time.time()})

        # drawing
        screen.fill((30, 30, 30))
        # draw resources
        for r in resources:
            color = (200, 180, 40) if not r.picked else (80, 80, 80)
            pygame.draw.circle(screen, color, (int(r.pos[0]), int(r.pos[1])), r.radius)
            if not r.picked:
                # draw label
                txt = font.render(f"{r.id}", True, (0, 0, 0))
                screen.blit(txt, (r.pos[0] - 6, r.pos[1] - 8))

        # draw agent
        ax, ay = int(agent.pos[0]), int(agent.pos[1])
        pygame.draw.circle(screen, (80, 200, 120), (ax, ay), AGENT_RADIUS)
        # draw current waypoints
        for wp in agent.waypoints:
            pygame.draw.circle(screen, (120, 180, 220), (int(wp[0]), int(wp[1])), 4)
            pygame.draw.line(screen, (60, 60, 60), (ax, ay), (int(wp[0]), int(wp[1])), 1)
            ax, ay = int(wp[0]), int(wp[1])

        # hud
        status = f"Agent state: {agent.state} | Waypoints: {len(agent.waypoints)} | Resources left: {len([r for r in resources if not r.picked])}"
        screen.blit(font.render(status, True, (220, 220, 220)), (8, 8))

        pygame.display.flip()

    agent.close()
    pygame.quit()


if __name__ == "__main__":
    # remove old log file
    try:
        if LOG_ACTIONS and os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
    except Exception:
        pass
    main()
