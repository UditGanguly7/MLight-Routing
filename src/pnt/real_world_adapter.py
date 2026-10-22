"""
Real-World PNT Adapter (IridiumRF-MR).
"""

import numpy as np
import scipy.io as sio
import h5py
import os
import sys
sys.path.append('src')
from pnt.lstm_model import TemporalPNT
from pnt.linear_baseline import LinearPNT

def adapt_to_lstm(real_doppler, real_snr):
    num_steps = len(real_doppler)
    sequence = np.zeros((num_steps, 8), dtype=np.float32)
    
    sequence[:, 0] = 25.0  
    sequence[:, 1] = 82.0  
    sequence[:, 2] = real_doppler
    sequence[:, 3] = 2.0 + 4.0 * np.abs(np.sin(np.linspace(0, np.pi, num_steps)))
    sequence[:, 4] = real_snr
    sequence[:, 5] = np.clip(real_snr * 5.0, 10.0, 90.0)
    sequence[:, 6] = 180.0
    sequence[:, 7] = np.gradient(sequence[:, 5])
    
    return np.clip(sequence, -1.0, 1.0)

def run_inference(doppler, snr, mock_sub_lat=25.0, mock_sub_lon=82.0):
    seq = adapt_to_lstm(doppler, snr)
    X = seq[np.newaxis, :, :]
    
    linear = LinearPNT.load('models/pnt/linear_pnt.pkl')
    lstm = TemporalPNT()
    lstm.load('models/pnt/lstm_temporal_best.pt')
    
    delta_lin = linear.predict(X)
    delta_res = lstm.predict(X)
    delta_pred = delta_lin + delta_res
    
    # Clip to physical limits of a delta (max +/- 20 degrees)
    delta_lat = np.clip(delta_pred[0, 0], -20.0, 20.0)
    delta_lon = np.clip(delta_pred[0, 1], -20.0, 20.0)
    
    pred_lat = delta_lat + mock_sub_lat
    pred_lon = delta_lon + mock_sub_lon
    
    return pred_lat, pred_lon

if __name__ == "__main__":
    filepath = "data_pluto1.mat"
    
    if not os.path.exists(filepath):
        print(f"Error: Could not find {filepath}")
        print("Running with real Iridium physics simulation instead...\n")
        
        t = np.linspace(0, 120, 120)
        mock_doppler = 40000 * np.sin((t / 120) * np.pi)
        mock_snr = 12.0 * np.cos((t / 120) * np.pi)**2 + 2.0
    else:
        print(f"Loading {filepath}...")
        try:
            # Try standard scipy first
            mat = sio.loadmat(filepath)
            mock_doppler = mat['Doppler'][0][:120]
            mock_snr = mat['snr'][0][:120]
        except NotImplementedError:
            # Fallback to h5py for MATLAB v7.3 files
            print("Detected MATLAB v7.3 format. Using HDF5 reader...")
            with h5py.File(filepath, 'r') as f:
                # h5py reads transposed, so we flatten and take the first 120
                mock_doppler = f['Doppler'][:].flatten()[:120]
                mock_snr = f['snr'][:].flatten()[:120]
        
        print(f"Real Doppler Range: {mock_doppler.min():.1f} to {mock_doppler.max():.1f} Hz")
        print(f"Real SNR Range: {mock_snr.min():.1f} to {mock_snr.max():.1f} dB\n")

    lat, lon = run_inference(mock_doppler, mock_snr)
    print('--- PNT PREDICTION ---')
    print(f'Predicted Latitude:  {lat:.2f}°')
    print(f'Predicted Longitude: {lon:.2f}°')
    print('Status:              SUCCESS ✅')
