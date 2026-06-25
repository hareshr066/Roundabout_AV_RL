"""
evaluation/demo_mode.py
====================================================================
PRESENTATION-QUALITY DEMO MODE FOR ROUNDABOUT AV RL

Features
--------
  * SUMO-GUI with automatic ego vehicle tracking
  * 0.25x slow motion (configurable)
  * Real-time colour-coded console telemetry panel
  * Pauses on MERGE event, SUCCESS, and COLLISION
  * Traffic density presets: LOW / MEDIUM / HIGH / VERY_HIGH
  * Compatible with any trained PPO model (.zip)

Usage
-----
    python evaluation/demo_mode.py [OPTIONS]

    Options:
      --model     PATH     Path to .zip PPO model  (default: auto-detect latest)
      --preset    NAME     Traffic preset: LOW | MEDIUM | HIGH | VERY_HIGH
                           (default: MEDIUM)
      --episodes  N        Number of demo episodes  (default: 5)
      --speed     FACTOR   Slow-motion factor 0.1-1.0  (default: 0.25)
      --no-pause           Disable event pauses (for recording)
      --demo-gui           Use gui-settings-demo.cfg (extra zoom)

Color Legend (SUMO-GUI)
-----------------------
  RED   = Ego vehicle (PPO agent)
  BLUE  = HDV (Human-Driven Vehicle)  -- stochastic Krauss model
  GREEN = AV  (Autonomous Vehicle)    -- deterministic IDM model
====================================================================
"""

import os
import sys
import time
import argparse
import glob
import numpy as np

# Ensure project root is on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from stable_baselines3 import PPO
from env.roundabout_env import RoundaboutEnv

# ----------------------------------------------------------------------
# ANSI colour helpers (safe fallback if terminal doesn't support them)
# ----------------------------------------------------------------------
def _enable_ansi():
    """Enable ANSI escape codes on Windows."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

_enable_ansi()

RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def clr(text, *codes):
    return "".join(codes) + str(text) + RESET


# ----------------------------------------------------------------------
# TRAFFIC PRESET DEFINITIONS
# ----------------------------------------------------------------------
PRESETS = {
    "LOW": {
        "route_file":   "sumo_network/routes_low.xml",
        "veh_per_hour": 100,
        "description":  "Light urban traffic -- easy merging",
        "hdv_ratio":    0.70,
    },
    "MEDIUM": {
        "route_file":   "sumo_network/routes_medium.xml",
        "veh_per_hour": 200,
        "description":  "Normal urban peak -- standard evaluation",
        "hdv_ratio":    0.60,
    },
    "HIGH": {
        "route_file":   "sumo_network/routes_high.xml",
        "veh_per_hour": 350,
        "description":  "Busy intersection -- high-difficulty evaluation",
        "hdv_ratio":    0.55,
    },
    "VERY_HIGH": {
        "route_file":   "sumo_network/routes_very_high.xml",
        "veh_per_hour": 500,
        "description":  "Near-capacity / congested -- stress test",
        "hdv_ratio":    0.50,
    },
}


# ----------------------------------------------------------------------
# MODEL AUTO-DETECTION
# ----------------------------------------------------------------------
def _auto_detect_model():
    search_dir = os.path.join(_ROOT, "results", "models")
    candidates = [
        c for c in glob.glob(os.path.join(search_dir, "*.zip"))
        if "val_test" not in os.path.basename(c) and "ablation" not in os.path.basename(c)
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


# ----------------------------------------------------------------------
# TELEMETRY PANEL
# ----------------------------------------------------------------------
def _print_telemetry(step, ep, total_ep, speed, accel, dist_to_entry,
                     ttc, gap, merge_state, reward, preset_name):
    """Print a real-time telemetry line to console."""
    # Colour-code TTC
    if ttc < 1.5:
        ttc_col = RED
    elif ttc < 3.0:
        ttc_col = YELLOW
    else:
        ttc_col = GREEN

    ttc_str = f"{ttc:.2f}s" if ttc < 999 else "inf  "

    # Speed bar (0-14 m/s, 15 chars)
    bar_len = min(15, int(speed / 14.0 * 15))
    speed_bar = "#" * bar_len + "." * (15 - bar_len)

    accel_col = GREEN if accel >= 0 else RED

    # Merge state colour
    state_colours = {
        "APPROACH":    CYAN,
        "MERGE_ZONE":  YELLOW,
        "CIRCULATING": GREEN,
        "EXITED":      GREEN,
    }
    state_col = state_colours.get(merge_state, WHITE)

    line = (
        f"  {DIM}[Ep{ep:2d}/{total_ep}|Stp{step:4d}]{RESET}"
        f"  Spd:{CYAN}{speed:5.2f}{RESET}m/s"
        f"  [{CYAN}{speed_bar}{RESET}]"
        f"  Acc:{accel_col}{accel:+5.2f}{RESET}"
        f"  TTC:{ttc_col}{ttc_str}{RESET}"
        f"  Gap:{gap:5.1f}m"
        f"  Dist:{dist_to_entry:5.1f}m"
        f"  {state_col}[{merge_state}]{RESET}"
        f"  Rew:{reward:+7.3f}"
        f"  {DIM}[{preset_name}]{RESET}"
    )
    print(line)


def _pause_for_event(label, duration=3.0, no_pause=False):
    """Display an event banner and optionally pause for keypress."""
    print("")
    print("  " + "=" * 58)
    print(f"  {BOLD}{YELLOW}  EVENT: {label}{RESET}")
    print("  " + "=" * 58)
    print("")
    if not no_pause:
        try:
            input(f"  {DIM}Press ENTER to continue...{RESET}")
        except (EOFError, KeyboardInterrupt):
            pass
    else:
        time.sleep(duration)


# ----------------------------------------------------------------------
# SUMMARY TABLE
# ----------------------------------------------------------------------
def _print_summary(episodes_data, model_name, preset_name):
    w = 85
    print("\n" + "=" * w)
    print(f"{'DEMO MODE -- EPISODE SUMMARY':^{w}}")
    print(f"{'Model: ' + model_name:^{w}}")
    print(f"{'Preset: ' + preset_name:^{w}}")
    print("=" * w)

    header = (f"{'Ep':<4}  {'Outcome':<12}  {'Steps':<7}  "
              f"{'T->Merge':>10}  {'AvgTTC':>9}  {'MinTTC':>9}  {'TotalRew':>10}")
    print(header)
    print("-" * w)

    successes = collisions = timeouts = 0
    for r in episodes_data:
        outcome = r["outcome"]
        if outcome == "SUCCESS":
            outcol = GREEN + BOLD
            successes += 1
        elif outcome == "COLLISION":
            outcol = RED + BOLD
            collisions += 1
        else:
            outcol = YELLOW
            timeouts += 1

        merge_str = f"{r['time_to_merge']:.2f}s" if isinstance(r["time_to_merge"], float) else "   N/A"
        avg_ttc   = f"{r['avg_ttc']:.2f}s"  if r["avg_ttc"] < 900 else "   N/A"
        min_ttc   = f"{r['min_ttc']:.2f}s"  if r["min_ttc"] < 900 else "   N/A"
        rew_str   = f"{r['total_reward']:+.2f}"

        print(f"{r['episode']:<4}  {outcol}{outcome:<12}{RESET}  {r['steps']:<7}  "
              f"{merge_str:>10}  {avg_ttc:>9}  {min_ttc:>9}  {rew_str:>10}")

    print("-" * w)
    n = len(episodes_data)
    sr = 100 * successes / n if n > 0 else 0
    cr = 100 * collisions / n if n > 0 else 0
    print(f"  {GREEN}Successes: {successes}/{n} ({sr:.0f}%){RESET}   "
          f"{RED}Collisions: {collisions}/{n} ({cr:.0f}%){RESET}   "
          f"{YELLOW}Timeouts: {timeouts}/{n}{RESET}")
    print("=" * w + "\n")


# ----------------------------------------------------------------------
# MERGE STATE HELPER
# ----------------------------------------------------------------------
def _get_merge_state(env):
    if env.entered_circulating:
        return "CIRCULATING"
    if env.reached_merge_zone:
        return "MERGE_ZONE"
    return "APPROACH"


# ----------------------------------------------------------------------
# MAIN DEMO RUNNER
# ----------------------------------------------------------------------
def run_demo(model_path, preset_name="MEDIUM", num_episodes=5,
             speed_factor=0.25, no_pause=False, demo_gui=False, hdv_ratio=None):

    preset = PRESETS[preset_name]
    active_hdv_ratio = hdv_ratio if hdv_ratio is not None else preset["hdv_ratio"]

    # Header
    print("\n" + "=" * 65)
    print(f"  {BOLD}{CYAN}ROUNDABOUT AV RL -- DEMO MODE{RESET}")
    print("=" * 65)
    print(f"  Model  : {os.path.basename(model_path)}")
    print(f"  Preset : {clr(preset_name, YELLOW, BOLD)}  --  {preset['description']}")
    print(f"  Flow   : ~{preset['veh_per_hour']} veh/hr/arm")
    print(f"  HDV Ratio: {active_hdv_ratio * 100:.0f}%")
    print(f"  Speed  : {speed_factor}x real-time")
    print(f"  Episodes: {num_episodes}")
    print(f"")
    print(f"  [RED]   = Ego (PPO agent)")
    print(f"  [BLUE]  = HDV (human driver, Krauss model)")
    print(f"  [GREEN] = AV  (autonomous, IDM model)")
    print("=" * 65 + "\n")

    # Load model
    print(f"Loading model: {model_path}")
    if not os.path.exists(model_path):
        print(f"{RED}[ERROR] Model not found: {model_path}{RESET}")
        sys.exit(1)
    model = PPO.load(model_path)
    print(f"{GREEN}[OK] Model loaded successfully.{RESET}\n")

    config_file = os.path.join(_ROOT, "sumo_network", "roundabout.sumocfg")

    episodes_data = []
    delay_per_step = speed_factor * 0.1  # step_length=0.1s
    zoom = 800 if demo_gui else 600

    try:
        for ep in range(num_episodes):
            print(f"\n{'=' * 60}")
            print(f"  {BOLD}EPISODE {ep + 1} / {num_episodes}  [{preset_name}]{RESET}")
            print(f"{'=' * 60}")

            # Create a fresh env (and SUMO-GUI) for every episode.
            # This prevents FatalTraCIError if the user closed SUMO-GUI
            # between episodes.
            env = RoundaboutEnv(
                config_file=config_file,
                gui=True,
                max_steps=800,          # 800 steps = 80s @ dt=0.1s (long arms need more time)
                fixed_hdv_ratio=active_hdv_ratio,
                fixed_spawn_distance=80.0,
                use_spatial_curriculum=False,
                traffic_density=preset_name.lower(),
                label=f"demo_ep{ep + 1}"
            )

            try:
                obs, info = env.reset()
            except Exception as e:
                print(f"\n  {RED}[ERROR] SUMO failed to start for episode {ep + 1}: {e}{RESET}")
                print(f"  {DIM}Skipping episode {ep + 1}.{RESET}")
                try:
                    env.close()
                except Exception:
                    pass
                continue

            # GUI: track ego and set zoom
            try:
                env.sim.conn.gui.trackVehicle("View #0", env.ego_id)
                env.sim.conn.gui.setZoom("View #0", zoom)
            except Exception:
                pass

            # Episode state
            done = False
            step = 0
            prev_speed = 0.0
            episode_ttcs = []
            min_ttc = float("inf")
            total_reward = 0.0
            time_to_merge_val = "N/A"

            merge_event_fired = False
            success_event_fired = False
            collision_event_fired = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)

                speed         = float(obs[0])
                dist_to_entry = float(obs[1])
                gap           = float(obs[4])

                obs, reward, terminated, truncated, step_info = env.step(action)
                done  = terminated or truncated
                step += 1
                total_reward += float(reward)

                # Instantaneous acceleration
                curr_speed = float(obs[0])
                inst_accel = (curr_speed - prev_speed) / env.dt
                prev_speed = curr_speed

                # TTC
                ttc = env._get_ttc_after_merge()
                if ttc < float("inf"):
                    ttc_display = min(ttc, 10.0)
                    episode_ttcs.append(ttc_display)
                    min_ttc = min(min_ttc, ttc_display)
                else:
                    ttc_display = 999.0

                merge_state = _get_merge_state(env)

                # Telemetry
                _print_telemetry(
                    step=step, ep=ep + 1, total_ep=num_episodes,
                    speed=speed, accel=inst_accel,
                    dist_to_entry=dist_to_entry, ttc=ttc_display,
                    gap=gap, merge_state=merge_state,
                    reward=float(reward), preset_name=preset_name,
                )

                # Event: Merge Zone
                if step_info.get("reached_merge_zone") and not merge_event_fired:
                    merge_event_fired = True
                    _pause_for_event(
                        "EGO ENTERED MERGE ZONE -- Watch gap acceptance!",
                        no_pause=no_pause
                    )

                # Event: Circulating lane
                if step_info.get("success") and not success_event_fired:
                    success_event_fired = True
                    time_to_merge_val = step_info.get("time_to_merge", step * env.dt)
                    print(f"\n  {BOLD}{GREEN}[SUCCESS] Ego entered circulating lane "
                          f"(T={time_to_merge_val:.2f}s){RESET}")
                    _pause_for_event(
                        "MERGE SUCCESS -- Ego is now in the circulating lane!",
                        no_pause=no_pause
                    )

                # Event: Collision
                if step_info.get("collision") and not collision_event_fired:
                    collision_event_fired = True
                    print(f"\n  {BOLD}{RED}[COLLISION] Ego vehicle crashed!{RESET}")
                    _pause_for_event(
                        "COLLISION -- Ego vehicle has crashed!",
                        duration=5.0, no_pause=no_pause
                    )

                # Unsafe TTC warning
                if ttc_display < env.ttc_threshold and ttc_display < 999.0:
                    print(f"  {RED}{BOLD}  !!! TTC CRITICAL: {ttc_display:.2f}s -- UNSAFE GAP !!!{RESET}")

                time.sleep(delay_per_step)

            # Episode outcome
            outcome = step_info.get("termination_reason", "timeout").upper()
            avg_ttc     = float(np.mean(episode_ttcs)) if episode_ttcs else 999.0
            min_ttc_val = min_ttc if min_ttc < float("inf") else 999.0

            if outcome == "SUCCESS":
                print(f"\n  {GREEN}{BOLD}** EPISODE {ep + 1}: SUCCESS **{RESET}")
                _pause_for_event(
                    f"EPISODE {ep + 1} SUCCESS -- Ego exited roundabout cleanly.",
                    no_pause=no_pause
                )
            elif outcome == "COLLISION":
                print(f"\n  {RED}{BOLD}EPISODE {ep + 1}: COLLISION{RESET}")
            else:
                print(f"\n  {YELLOW}EPISODE {ep + 1}: TIMEOUT{RESET}")

            print(f"  Total Reward: {total_reward:+.2f}  |  "
                  f"Avg TTC: {avg_ttc:.2f}s  |  Min TTC: {min_ttc_val:.2f}s")

            episodes_data.append({
                "episode":       ep + 1,
                "outcome":       outcome,
                "steps":         step,
                "time_to_merge": time_to_merge_val,
                "avg_ttc":       avg_ttc,
                "min_ttc":       min_ttc_val,
                "total_reward":  total_reward,
            })

            print(f"\n  {DIM}Pausing before next episode -- waiting for SUMO-GUI to close...{RESET}")
            try:
                env.close()
            except Exception:
                pass
            # Grace period: give SUMO-GUI time to fully terminate before
            # we launch a new SUMO process for the next episode.
            time.sleep(6.0)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Demo interrupted by user (Ctrl+C).{RESET}")

    _print_summary(episodes_data, os.path.basename(model_path), preset_name)


# ----------------------------------------------------------------------
# CLI ENTRY POINT
# ----------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Presentation-quality demo mode for Roundabout AV RL"
    )
    parser.add_argument("--model",    type=str,   default=None,
                        help="Path to trained PPO model (.zip). Auto-detects latest if not given.")
    parser.add_argument("--preset",   type=str,   default="MEDIUM",
                        choices=list(PRESETS.keys()),
                        help="Traffic density preset (default: MEDIUM)")
    parser.add_argument("--episodes", type=int,   default=5,
                        help="Number of demo episodes (default: 5)")
    parser.add_argument("--speed",    type=float, default=0.25,
                        help="Slow-motion factor 0.1-1.0 (default: 0.25)")
    parser.add_argument("--no-pause", action="store_true",
                        help="Disable interactive pauses (for automated recording)")
    parser.add_argument("--demo-gui", action="store_true",
                        help="Use higher zoom (800) for presentation mode")
    parser.add_argument("--hdv-ratio", type=float, default=None,
                        help="Override HDV ratio (0.0 to 1.0) dynamically. If not set, preset's default ratio is used.")
    return parser.parse_args()


def main():
    args = _parse_args()

    # Resolve model path
    if args.model:
        model_path = args.model
    else:
        model_path = _auto_detect_model()
        if model_path is None:
            print(f"{RED}[ERROR] No .zip model found in results/models/.{RESET}")
            print("  Please specify --model PATH or train a model first.")
            sys.exit(1)
        print(f"{CYAN}Auto-detected model: {model_path}{RESET}")

    preset_name = args.preset.upper()
    speed = max(0.05, min(1.0, args.speed))

    run_demo(
        model_path=model_path,
        preset_name=preset_name,
        num_episodes=args.episodes,
        speed_factor=speed,
        no_pause=args.no_pause,
        demo_gui=args.demo_gui,
        hdv_ratio=args.hdv_ratio,
    )


if __name__ == "__main__":
    main()
