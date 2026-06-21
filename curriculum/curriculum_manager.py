import logging
from collections import deque

# Configure logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [Curriculum] - %(levelname)s - %(message)s")

class CurriculumManager:
    """
    Manages curriculum stages during reinforcement learning training.
    Progressively increases the ratio of Human-Driven Vehicles (HDVs) to Autonomous Vehicles (AVs)
    to introduce environmental uncertainty once the agent masters simpler stages.
    """
    def __init__(self, target_success_rate=0.85, window_size=100):
        # 5 curriculum stages matching the percentage profiles
        self.stages = {
            1: {"hdv_ratio": 0.00, "description": "Stage 1: 0% HDV / 100% AV (Deterministic, Cooperative traffic)"},
            2: {"hdv_ratio": 0.25, "description": "Stage 2: 25% HDV / 75% AV (Slight uncertainty)"},
            3: {"hdv_ratio": 0.50, "description": "Stage 3: 50% HDV / 50% AV (Mixed-autonomy equilibrium)"},
            4: {"hdv_ratio": 0.75, "description": "Stage 4: 75% HDV / 25% AV (High uncertainty, dominant human traffic)"},
            5: {"hdv_ratio": 1.00, "description": "Stage 5: 100% HDV / 0% AV (Maximum uncertainty, standard human traffic)"}
        }
        
        self.current_stage = 1
        self.target_success_rate = target_success_rate
        self.window_size = window_size
        
        # Performance tracking metrics
        self.episode_history = deque(maxlen=window_size)  # Stores True (success) or False (failure)
        self.total_episodes = 0
        self.total_successes = 0
        
        logging.info(f"Initialized Curriculum Manager. Starting at Stage 1 (HDV Ratio: 0.0).")

    @property
    def current_hdv_ratio(self):
        """Returns the ratio of human-driven vehicles for the active curriculum stage."""
        return self.stages[self.current_stage]["hdv_ratio"]

    @property
    def current_description(self):
        """Returns the description of the active curriculum stage."""
        return self.stages[self.current_stage]["description"]

    @property
    def rolling_success_rate(self):
        """Computes the success rate over the sliding window (last 'window_size' episodes)."""
        if len(self.episode_history) == 0:
            return 0.0
        return sum(self.episode_history) / len(self.episode_history)

    def update_metrics(self, success: bool):
        """
        Updates the episode history, total counts, and checks if the agent qualifies for stage advancement.
        
        Args:
            success (bool): True if the agent reached the target exit safely, False otherwise.
            
        Returns:
            bool: True if a stage transition occurred, False otherwise.
        """
        self.total_episodes += 1
        if success:
            self.total_successes += 1
            
        self.episode_history.append(success)
        
        # Check for advancement conditions
        # Only advance if we have collected enough episodes to fill the sliding window size
        if len(self.episode_history) >= self.window_size:
            current_rate = self.rolling_success_rate
            if current_rate > self.target_success_rate:
                return self.advance_stage()
                
        return False

    def advance_stage(self):
        """Advances the training to the next curriculum stage if not already at maximum stage."""
        if self.current_stage < len(self.stages):
            old_stage = self.current_stage
            self.current_stage += 1
            
            # Clear rolling window history so the agent has time to adapt to the new stage's dynamics
            # before we evaluate performance for the next stage transition.
            self.episode_history.clear()
            
            logging.info(
                f"\n=== CURRICULUM ADVANCEMENT ==="
                f"\nAdvanced from Stage {old_stage} to Stage {self.current_stage}!"
                f"\nNew HDV Ratio: {self.current_hdv_ratio * 100:.0f}%"
                f"\nDescription: {self.current_description}"
                f"\n=============================="
            )
            return True
        else:
            logging.info("Agent is already at maximum curriculum stage (Stage 5). Training profile remains unchanged.")
            return False
            
    def get_status(self):
        """Returns a formatted status dict for monitoring."""
        return {
            "current_stage": self.current_stage,
            "hdv_ratio": self.current_hdv_ratio,
            "rolling_success_rate": self.rolling_success_rate,
            "window_fill": len(self.episode_history),
            "total_episodes": self.total_episodes
        }
