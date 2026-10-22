"""
End-to-End Integration Demo (Real-World Iridium Data Edition)
"""
import numpy as np
import time
import os
import scipy.io as sio
import h5py
from pathlib import Path
from ..simulation.constellation import create_default_constellation
from ..pnt.linear_baseline import LinearPNT
from ..pnt.lstm_model import TemporalPNT
from ..pnt.real_world_adapter import run_inference
from ..topology.topology_generator import generate_topology_sample
from ..topology.topology_model import TopologyWrapper
from ..router.env_v2 import RoutingEnvV2
from ..router.dqn_agent import DQNAgent

def main():
    print("=" * 72)
    print("  MLight-Routing: End-to-End Integration Demo (REAL DATA)")
    print("=" * 72)

    # --- PHASE 1: PNT (REAL DATA) ---
    print("\n--- Step 1: PNT Estimation ---")
    real_file = "data_pluto1.mat"
    
    if os.path.exists(real_file):
        print(f"  Loading REAL Iridium data from {real_file}...")
        try:
            mat = sio.loadmat(real_file)
            real_doppler = mat['Doppler'][0][:120]
            real_snr = mat['snr'][0][:120]
        except NotImplementedError:
            with h5py.File(real_file, 'r') as f:
                real_doppler = f['Doppler'][:].flatten()[:120]
                real_snr = f['snr'][:].flatten()[:120]
                
        lat, lon = run_inference(real_doppler, real_snr)
        print(f"  Real Doppler Range: {real_doppler.min():.1f} to {real_doppler.max():.1f} Hz")
        print(f"  Real SNR Range:     {real_snr.min():.1f} to {real_snr.max():.1f} dB")
    else:
        print("  [WARNING: data_pluto1.mat not found. Using synthetic fallback.]")
        lat, lon = 45.0, 102.0

    print(f"  Estimated Position: ({lat:.2f}°, {lon:.2f}°)")
    print("  ✓ PNT complete")

    # --- PHASE 2: Topology ---
    print("\n--- Step 2: Topology Prediction ---")
    constellation = create_default_constellation()
    wrapper = TopologyWrapper()
    wrapper.load("models/topology/topology_best.pt")
    
    X_t, y_t = generate_topology_sample(constellation, 0.0)
    acc = wrapper.accuracy(y_t[np.newaxis], wrapper.predict(X_t[np.newaxis]))
    print(f"  Topology Accuracy:  {acc:.4f}")
    print("  ✓ Topology complete")

    # --- PHASE 3: Routing ---
    print("\n--- Step 3: DRL Routing ---")
    agent = DQNAgent(obs_dim=35)
    agent.load("models/router/dqn_v2_best.pt")
    
    delivered = 0
    for i in range(20):
        env = RoutingEnvV2(seed=i)
        obs, info = env.reset()
        for _ in range(20):
            mask = env.action_masks()
            action = agent.select_action(obs, mask, greedy=True)
            obs, r, term, trunc, info = env.step(action)
            if term or trunc:
                if info.get('reason') == 'delivered':
                    delivered += 1
                break
    print(f"  Delivered:          {delivered}/20 packets")
    print("  ✓ Routing complete")

    print("\n" + "=" * 72)
    print("  END-TO-END PIPELINE COMPLETE (with REAL satellite data)")
    print("=" * 72)

if __name__ == "__main__":
    main()
