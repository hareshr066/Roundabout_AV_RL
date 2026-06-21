import os
import subprocess
import sys

def build_roundabout_network():
    """
    Build the SUMO roundabout network using netconvert.

    GEOMETRY (Updated for Visual Realism Audit):
    ─────────────────────────────────────────────
    Ring radius:     30 m  (60 m inscribed circle diameter)
                     → Realistic single-lane urban roundabout per FHWA / UK DMRB
                     → Previous value was 20 m (mini-roundabout / parking lot scale)

    Entry arm length: ~170 m  (far nodes at ±200, ring nodes at ±30)
                     → Previous value was ~80 m (vehicles reached merge in ~6 s)
                     → 170 m gives ~12 s realistic approach at 50 km/h

    Ring speed:      8.33 m/s = 30 km/h  (standard urban roundabout)
                     → Previous was 11.11 m/s = 40 km/h (too fast for size)

    Lane width:      4.0 m for circulating lanes  (realistic, previously unspecified)

    Quadrant midpoints (45° on 30 m radius):
        x = y = 30 × cos(45°) = 30 × 0.70711 = 21.21 m
    """
    # File names (written relative to script's directory)
    nod_file = "roundabout.nod.xml"
    edg_file = "roundabout.edg.xml"
    net_file = "roundabout.net.xml"

    # ────────────────────────────────────────────────────────────────────
    # 1. NODE DEFINITIONS — Realistic 30 m radius urban roundabout
    # ────────────────────────────────────────────────────────────────────
    nodes_content = """<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    <!-- Roundabout ring junction nodes (30 m radius, 60 m inscribed circle) -->
    <node id="N" x="0"    y="30"   type="priority"/>
    <node id="E" x="30"   y="0"    type="priority"/>
    <node id="S" x="0"    y="-30"  type="priority"/>
    <node id="W" x="-30"  y="0"    type="priority"/>

    <!-- Far boundary nodes: entry/exit arms of approx 170 m each -->
    <node id="N_far" x="0"    y="200"   type="priority"/>
    <node id="E_far" x="200"  y="0"     type="priority"/>
    <node id="S_far" x="0"    y="-200"  type="priority"/>
    <node id="W_far" x="-200" y="0"     type="priority"/>
</nodes>
"""

    # 2. EDGE DEFINITIONS
    edges_content = """<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    <!-- Circular Ring: 1 lane, 30 km/h (8.33 m/s), 4.0 m wide -->
    <!-- Shape midpoints at 45-deg on 30m radius: 30*cos(45)=21.21 m -->
    <edge id="circ_S_E" from="S" to="E" numLanes="1" speed="8.33" shape="0,-30 21.21,-21.21 30,0"/>
    <edge id="circ_E_N" from="E" to="N" numLanes="1" speed="8.33" shape="30,0 21.21,21.21 0,30"/>
    <edge id="circ_N_W" from="N" to="W" numLanes="1" speed="8.33" shape="0,30 -21.21,21.21 -30,0"/>
    <edge id="circ_W_S" from="W" to="S" numLanes="1" speed="8.33" shape="-30,0 -21.21,-21.21 0,-30"/>

    <!-- South Arm (~170 m, 50 km/h) -->
    <edge id="entry_S" from="S_far" to="S" numLanes="1" speed="13.89"/>
    <edge id="exit_S"  from="S" to="S_far" numLanes="1" speed="13.89"/>

    <!-- East Arm (~170 m, 50 km/h) -->
    <edge id="entry_E" from="E_far" to="E" numLanes="1" speed="13.89"/>
    <edge id="exit_E"  from="E" to="E_far" numLanes="1" speed="13.89"/>

    <!-- North Arm (~170 m, 50 km/h) -->
    <edge id="entry_N" from="N_far" to="N" numLanes="1" speed="13.89"/>
    <edge id="exit_N"  from="N" to="N_far" numLanes="1" speed="13.89"/>

    <!-- West Arm (~170 m, 50 km/h) -->
    <edge id="entry_W" from="W_far" to="W" numLanes="1" speed="13.89"/>
    <edge id="exit_W"  from="W" to="W_far" numLanes="1" speed="13.89"/>
</edges>
"""

    print("=" * 60)
    print("   ROUNDABOUT NETWORK BUILDER - Visual Realism Edition")
    print("=" * 60)
    print(f"  Ring radius:     30 m  (60 m inscribed circle diameter)")
    print(f"  Arm length:     ~170 m  (far nodes at +/-200 m)")
    print(f"  Ring speed:      8.33 m/s  (30 km/h)")
    print(f"  Arm speed:      13.89 m/s  (50 km/h)")
    print(f"  Lane width:      4.0 m  (circulating lanes)")
    print("=" * 60)

    print("\nWriting node and edge definition files...")
    with open(nod_file, "w", encoding="utf-8") as f:
        f.write(nodes_content)
    with open(edg_file, "w", encoding="utf-8") as f:
        f.write(edges_content)

    print("Compiling network using netconvert...")
    print("  (--roundabouts.guess=true ensures correct right-of-way)")
    try:
        cmd = [
            "netconvert",
            f"--node-files={nod_file}",
            f"--edge-files={edg_file}",
            "--roundabouts.guess=true",
            "--default.lanewidth=4.0",     # 4 m realistic circulating lane width
            f"--output-file={net_file}",
            "--no-warnings=false"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout:
            print("netconvert stdout:")
            print(result.stdout)
        if result.stderr:
            print("netconvert stderr/warnings:")
            print(result.stderr)
        print(f"\n[OK] Successfully generated: {net_file}")
        print(f"  Network file size: {os.path.getsize(net_file):,} bytes")

    except FileNotFoundError:
        print("\n[ERROR] netconvert not found in PATH.", file=sys.stderr)
        print("  Make sure SUMO is installed and SUMO_HOME is set.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] netconvert failed: {e}", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Change to script directory so all paths are relative
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    build_roundabout_network()
