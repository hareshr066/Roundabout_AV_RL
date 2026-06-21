import os
import sys
import time

# Ensure the root project directory is in the python path for importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.traci_connection import TraCIConnection

def run_control_test():
    config_file = os.path.join("sumo_network", "roundabout.sumocfg")
    
    print(f"Connecting to: {config_file}")
    
    # Initialize connection manager (headless mode)
    manager = TraCIConnection(
        config_path=config_file,
        gui=False,
        step_length=0.1,
        label="control_test_client"
    )
    
    try:
        # 1. Start simulation
        print("\n[Step 1] Starting SUMO simulation...")
        conn = manager.start()
        
        # Spawn a vehicle
        # Route 'r_N_S' is defined in routes.xml (North Entry -> West Circle -> South Exit)
        veh_id = "test_car"
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
        
        # Override default SUMO speed safety behaviors
        # SpeedMode: 0 = Disable all automatic safety/car-following logic. Speed is controlled 100% manually.
        # LaneChangeMode: 0 = Disable automatic lane changing.
        conn.vehicle.setSpeedMode(veh_id, 0)
        conn.vehicle.setLaneChangeMode(veh_id, 0)
        
        # Advance 1 step to physically insert the vehicle into the lane
        manager.step()
        
        print("\n[Step 3] Beginning step-by-step control loop...")
        print("-" * 80)
        print(f"{'Step':<6} | {'Target Speed':<14} | {'Current Speed':<15} | {'Lane ID':<10} | {'Lane Position':<14} | {'Odometer':<10}")
        print("-" * 80)
        
        # Loop for 30 steps (3.0 seconds of simulated time)
        for step in range(30):
            # Read vehicle telemetry using our connection helpers
            speed = manager.get_vehicle_speed(veh_id)
            pos_x, pos_y = manager.get_vehicle_position(veh_id)
            lane_pos = manager.get_lane_position(veh_id)
            lane_id = manager.get_lane_id(veh_id)
            distance = manager.get_distance_traveled(veh_id)
            
            # Print current state
            print(f"{step+1:<6} | {'-':<14} | {speed:<15.4f} | {lane_id:<10} | {lane_pos:<14.4f} | {distance:<10.2f}")
            
            # Actively command changes in speed/acceleration through Python
            if step == 5:
                # Command target speed directly (10 m/s)
                target = 10.0
                print(f">>>> Python command: Set vehicle speed to {target} m/s")
                manager.set_vehicle_speed(veh_id, target)
                
            elif step == 15:
                # Apply positive acceleration (2.0 m/s^2)
                accel = 2.0
                print(f">>>> Python command: Apply acceleration of {accel} m/s^2")
                manager.apply_acceleration(veh_id, accel)
                
            elif step == 20:
                # Apply hard deceleration (-4.0 m/s^2)
                decel = -4.0
                print(f">>>> Python command: Apply deceleration of {decel} m/s^2")
                manager.apply_acceleration(veh_id, decel)
                
            # Advance simulation step
            manager.step()
            
        print("-" * 80)
        print("\nTraCI state reading and control test completed successfully!")
        
    except Exception as e:
        print(f"\nTraCI control test FAILED with error: {e}", file=sys.stderr)
        raise e
        
    finally:
        # Clean up connection
        print("\nClosing TraCI connection...")
        manager.close()
        print("Connection closed safely.")

if __name__ == "__main__":
    run_control_test()
