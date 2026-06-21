import logging
from collections import deque

# Configure logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [SpatialCurriculum] - %(levelname)s - %(message)s")

class SpatialCurriculumManager:
    """
    Manages spatial curriculum stages during reinforcement learning training.
    Progressively increases the spawn distance from the roundabout entry (15m -> 30m -> 50m -> 80m)
    to allow the agent to learn merging before learning the long-distance approach.
    """
    def __init__(self, target_success_rate=0.80, window_size=50):
        # 4 curriculum stages for spawn distance
        self.stages = {
            1: {"spawn_distance": 15.0, "description": "Stage A: Spawn 15 m from roundabout entry"},
            2: {"spawn_distance": 30.0, "description": "Stage B: Spawn 30 m from roundabout entry"},
            3: {"spawn_distance": 50.0, "description": "Stage C: Spawn 50 m from roundabout entry"},
            4: {"spawn_distance": 80.0, "description": "Stage D: Spawn 80 m from roundabout entry"}
        }
        
        self.current_stage = 1
        self.target_success_rate = target_success_rate
        self.window_size = window_size
        
        # Rolling histories for logging and decision making
        self.episode_history = deque(maxlen=window_size)        # Successes (True/False)
        self.merge_reach_history = deque(maxlen=window_size)    # Reached merge zone (True/False)
        self.merge_attempt_history = deque(maxlen=window_size)  # Attempted merge (True/False)
        self.collision_history = deque(maxlen=window_size)      # Collisions (True/False)
        
        self.total_episodes = 0
        self.total_successes = 0
        
        logging.info(f"Initialized Spatial Curriculum Manager. Starting at Stage A (Spawn Distance: 15.0 m).")

    @property
    def current_spawn_distance(self):
        """Returns the spawn distance for the active curriculum stage."""
        return self.stages[self.current_stage]["spawn_distance"]

    @property
    def current_description(self):
        """Returns the description of the active curriculum stage."""
        return self.stages[self.current_stage]["description"]

    @property
    def rolling_success_rate(self):
        """Computes the success rate over the sliding window."""
        if len(self.episode_history) == 0:
            return 0.0
        return sum(self.episode_history) / len(self.episode_history)

    @property
    def rolling_merge_reach_rate(self):
        """Computes the merge zone reach rate over the sliding window."""
        if len(self.merge_reach_history) == 0:
            return 0.0
        return sum(self.merge_reach_history) / len(self.merge_reach_history)

    @property
    def rolling_merge_attempt_rate(self):
        """Computes the merge attempt rate over the sliding window."""
        if len(self.merge_attempt_history) == 0:
            return 0.0
        return sum(self.merge_attempt_history) / len(self.merge_attempt_history)

    @property
    def rolling_collision_rate(self):
        """Computes the collision rate over the sliding window."""
        if len(self.collision_history) == 0:
            return 0.0
        return sum(self.collision_history) / len(self.collision_history)

    def update_metrics(self, success: bool, reached_merge: bool, attempted_merge: bool, collision: bool):
        """
        Updates the episode statistics and checks for stage advancement.
        """
        self.total_episodes += 1
        if success:
            self.total_successes += 1
            
        self.episode_history.append(success)
        self.merge_reach_history.append(reached_merge)
        self.merge_attempt_history.append(attempted_merge)
        self.collision_history.append(collision)
        
        # Check advancement conditions
        if len(self.episode_history) >= self.window_size:
            current_rate = self.rolling_success_rate
            if current_rate > self.target_success_rate:
                return self.advance_stage()
                
        return False

    def advance_stage(self):
        """Advances to the next spatial curriculum stage if not already at maximum stage (Stage 4)."""
        if self.current_stage < len(self.stages):
            old_stage = self.current_stage
            self.current_stage += 1
            
            # Clear histories to allow learning adaptation
            self.episode_history.clear()
            self.merge_reach_history.clear()
            self.merge_attempt_history.clear()
            self.collision_history.clear()
            
            logging.info(
                f"\n=== SPATIAL CURRICULUM ADVANCEMENT ==="
                f"\nAdvanced from Stage {old_stage} to Stage {self.current_stage}!"
                f"\nNew Spawn Distance: {self.current_spawn_distance:.1f} m"
                f"\nDescription: {self.current_description}"
                f"\n======================================="
            )
            return True
        else:
            logging.info("Agent is already at maximum spatial curriculum stage (Stage D).")
            return False

    def get_status(self):
        """Returns status dictionary."""
        return {
            "current_stage": self.current_stage,
            "spawn_distance": self.current_spawn_distance,
            "rolling_success_rate": self.rolling_success_rate,
            "rolling_merge_reach_rate": self.rolling_merge_reach_rate,
            "rolling_merge_attempt_rate": self.rolling_merge_attempt_rate,
            "rolling_collision_rate": self.rolling_collision_rate,
            "window_fill": len(self.episode_history),
            "total_episodes": self.total_episodes
        }
