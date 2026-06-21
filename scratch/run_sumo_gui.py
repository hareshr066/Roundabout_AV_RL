import os
import sys
import time

# Ensure the root project directory is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.traci_connection import TraCIConnection

def run_gui_simulation():
    config_file = os.path.join("sumo_network", "roundabout.sumocfg")
    
    print(f"Connecting to SUMO-GUI using: {config_file}")
    
    # Initialize connection manager with GUI mode enabled
    manager = TraCIConnection(
        config_path=config_file,
        gui=True,
        step_length=0.1,
        label="gui_simulation_run"
    )
    
    try:
        print("\n[Step 1] Starting SUMO-GUI...")
        conn = manager.start()
        print("SUMO-GUI started and TraCI connected successfully.")
        
        veh_id = "ego_car"
        route_id = "r_N_S"
        vtype_id = "passenger_car"
        
        print(f"\n[Step 2] Spawning vehicle '{veh_id}' on route '{route_id}'...")
        conn.vehicle.add(
            vehID=veh_id,
            routeID=route_id,
            typeID=vtype_id,
            departLane="0",
            departPos="0.0",
            departSpeed="0.0"
        )
        
        # Override default safety behaviors to allow full speed control
        conn.vehicle.setSpeedMode(veh_id, 0)
        conn.vehicle.setLaneChangeMode(veh_id, 0)
        
        # Advance 1 step to spawn the vehicle
        manager.step()
        
        # Track the vehicle in the GUI and set zoom
        try:
            conn.gui.trackVehicle("View #0", veh_id)
            conn.gui.setZoom("View #0", 150.0)
        except Exception:
            pass
        
        print("\n[Step 3] Running simulation loop (300 steps)...")
        print("-" * 80)
        print(f"{'Step':<6} | {'Current Speed':<15} | {'Lane ID':<10} | {'Lane Position':<14} | {'Odometer':<10}")
        print("-" * 80)
        
        for step in range(300):
            # Read vehicle telemetry
            speed = manager.get_vehicle_speed(veh_id)
            lane_pos = manager.get_lane_position(veh_id)
            lane_id = manager.get_lane_id(veh_id)
            distance = manager.get_distance_traveled(veh_id)
            
            # Print current state
            print(f"{step+1:<6} | {speed:<15.4f} | {lane_id:<10} | {lane_pos:<14.4f} | {distance:<10.2f}")
            
            # Accelerate/maintain vehicle speed
            if step < 80:
                manager.set_vehicle_speed(veh_id, min(speed + 0.2, 10.0))
            elif step < 180:
                # Maintain constant speed
                manager.set_vehicle_speed(veh_id, 10.0)
            else:
                manager.set_vehicle_speed(veh_id, max(speed - 0.2, 0.0))
                
            # Advance simulation step
            manager.step()
            
            # Pause slightly so the GUI is easily watchable
            time.sleep(0.15)
            
        print("-" * 80)
        print("\nSUMO-GUI simulation loop finished successfully!")
        
        print("\nSimulation finished. Keeping SUMO-GUI open for 60 seconds so you can inspect...")
        for i in range(60, 0, -1):
            print(f"Closing in {i:2d} seconds...", end="\r")
            time.sleep(1)
        print()
        
    except Exception as e:
        print(f"\nSimulation run FAILED with error: {e}", file=sys.stderr)
        raise e
        
    finally:
        print("\nClosing TraCI connection and SUMO-GUI...")
        manager.close()
        print("Closed safely.")

if __name__ == "__main__":
    run_gui_simulation()
