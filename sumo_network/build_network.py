import os
import subprocess
import sys

def build_roundabout_network():
    # Define file names
    nod_file = "roundabout.nod.xml"
    edg_file = "roundabout.edg.xml"
    net_file = "roundabout.net.xml"
    
    # 1. Write the nodes (junctions) file
    nodes_content = """<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    <!-- Roundabout Central Circle Nodes -->
    <node id="N" x="0" y="25" type="priority"/>
    <node id="E" x="25" y="0" type="priority"/>
    <node id="S" x="0" y="-25" type="priority"/>
    <node id="W" x="-25" y="0" type="priority"/>
    
    <!-- Far-end boundary nodes (origins/destinations) -->
    <node id="N_far" x="0" y="150" type="priority"/>
    <node id="E_far" x="150" y="0" type="priority"/>
    <node id="S_far" x="0" y="-150" type="priority"/>
    <node id="W_far" x="-150" y="0" type="priority"/>
</nodes>
"""
    
    # 2. Write the edges (roads) file
    # We specify shape points using the radius (25m) to make the central segments circular.
    # Radius = 25m. At 45 degrees: 25 * cos(45) = 17.68, 25 * sin(45) = 17.68.
    edges_content = """<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    <!-- Circular Ring (2 lanes, speed 11.11 m/s = 40 km/h) -->
    <edge id="circ_S_E" from="S" to="E" numLanes="2" speed="11.11" shape="0,-25 17.68,-17.68 25,0"/>
    <edge id="circ_E_N" from="E" to="N" numLanes="2" speed="11.11" shape="25,0 17.68,17.68 0,25"/>
    <edge id="circ_N_W" from="N" to="W" numLanes="2" speed="11.11" shape="0,25 -17.68,17.68 -25,0"/>
    <edge id="circ_W_S" from="W" to="S" numLanes="2" speed="11.11" shape="-25,0 -17.68,-17.68 0,-25"/>

    <!-- South Arm (2 lanes entering, 2 lanes exiting, speed 13.89 m/s = 50 km/h) -->
    <edge id="entry_S" from="S_far" to="S" numLanes="2" speed="13.89"/>
    <edge id="exit_S" from="S" to="S_far" numLanes="2" speed="13.89"/>

    <!-- East Arm (2 lanes entering, 2 lanes exiting, speed 13.89 m/s = 50 km/h) -->
    <edge id="entry_E" from="E_far" to="E" numLanes="2" speed="13.89"/>
    <edge id="exit_E" from="E" to="E_far" numLanes="2" speed="13.89"/>

    <!-- North Arm (2 lanes entering, 2 lanes exiting, speed 13.89 m/s = 50 km/h) -->
    <edge id="entry_N" from="N_far" to="N" numLanes="2" speed="13.89"/>
    <edge id="exit_N" from="N" to="N_far" numLanes="2" speed="13.89"/>

    <!-- West Arm (2 lanes entering, 2 lanes exiting, speed 13.89 m/s = 50 km/h) -->
    <edge id="entry_W" from="W_far" to="W" numLanes="2" speed="13.89"/>
    <edge id="exit_W" from="W" to="W_far" numLanes="2" speed="13.89"/>
</edges>
"""

    print("Writing node and edge definition files...")
    with open(nod_file, "w") as f:
        f.write(nodes_content)
    with open(edg_file, "w") as f:
        f.write(edges_content)
        
    print("Compiling network using netconvert...")
    # Run netconvert with automatic roundabout guessing enabled
    # This automatically sets correct right-of-way (circulating traffic has priority)
    try:
        cmd = [
            "netconvert",
            f"--node-files={nod_file}",
            f"--edge-files={edg_file}",
            "--roundabouts.guess=true",
            f"--output-file={net_file}",
            "--no-warnings=false"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("netconvert output:")
        print(result.stdout)
        if result.stderr:
            print("netconvert errors/warnings:")
            print(result.stderr)
        print(f"Successfully generated network file: {net_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error compiling network: {e}", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Change working directory to the script's directory so paths are relative
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    build_roundabout_network()
