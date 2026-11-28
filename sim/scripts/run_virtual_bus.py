"""Start the virtual SO-101 servo bus: a serial device at /tmp/so101_bus, driving MuJoCo.

  python sim/scripts/run_virtual_bus.py --viewer                  # arm in MuJoCo, watch it
  python sim/scripts/run_virtual_bus.py --standalone              # servos only, no MuJoCo
  python sim/scripts/run_virtual_bus.py --latency-ms 2 --drop-rate 0.01   # a bad bus
  python sim/scripts/run_virtual_bus.py --start ready             # start in 'ready' (default: park)

The arm starts like a real one after power-up: torque OFF. In 'park' it rests folded on its
joint stops; started in 'ready' it sinks under gravity until a driver switches torque on.

Then, in another terminal, any program can open /tmp/so101_bus like a real /dev/ttyACM0,
for example:  python sim/scripts/bus_client_demo.py
Stop with Ctrl-C.
"""
import argparse
import time

from so101_sim.feetech.emulator import FeetechBusEmulator
from so101_sim.feetech.mujoco_arm import ServoArm
from so101_sim.feetech.virtual_bus import VirtualSerialBus
from so101_sim.scene import PARK_Q, ready_q


def main(a):
    viewer = None
    if a.standalone:
        target = FeetechBusEmulator()
    else:
        target = ServoArm(q0=ready_q() if a.start == "ready" else PARK_Q)
        if a.viewer:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(target.m, target.d)
    vb = VirtualSerialBus(target, link=a.link, extra_latency_s=a.latency_ms / 1000,
                          drop_rate=a.drop_rate, on_tick=viewer.sync if viewer else None)
    vb.start()
    print(f"virtual Feetech bus ready: {vb.device} -> {vb.path}  "
          f"({'standalone servos' if a.standalone else 'MuJoCo SO-101'}, ids 1-6, 1 Mbps)")
    print("open it from another terminal; Ctrl-C to stop")
    try:
        last = dict(vb.stats)
        while viewer is None or viewer.is_running():
            time.sleep(2.0)
            s = vb.stats
            rate = (s["requests"] - last["requests"]) / 2.0
            print(f"  {rate:6.0f} requests/s   replies {s['replies']}   dropped {s['dropped']}   "
                  f"physics lag {s['lag_s'] * 1e3:.1f} ms")
            last = dict(s)
    except KeyboardInterrupt:
        pass
    finally:
        vb.stop()
        if viewer:
            viewer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--link", default="/tmp/so101_bus")
    p.add_argument("--standalone", action="store_true", help="servo physics only, no MuJoCo arm")
    p.add_argument("--viewer", action="store_true", help="show the MuJoCo arm")
    p.add_argument("--start", choices=["park", "ready"], default="park", help="initial arm pose")
    p.add_argument("--latency-ms", type=float, default=0.0, help="extra reply delay")
    p.add_argument("--drop-rate", type=float, default=0.0, help="fraction of requests left unanswered")
    main(p.parse_args())
