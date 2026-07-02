
# Event Detection for Seismic Data

This repository contains the implementation of a deep learning framework designed for automated Earthquake Event Detection in multivariate, 3-channel seismological time-series data. The project explores the transition from traditional anomaly detection to supervised event boundary delineation under real-world, noisy conditions and hardware constraints.


# Model Architecture

1. 1D-CNN Layers: Act as adaptive frequency filters to extract spatial features and local spectral characteristics from raw waveforms. 
First Filter -> 
Wide filter for the context. Second Filter -> Tight Filter for details

2. Bidirectional LSTM (BiLSTM): Captures temporal dependencies by processing the time series in both forward and backward directions, ensuring precise identification of event onset and offset boundaries (including the coda wave phase).

3. Linear Layer: The features from LSTM pass from an Linear Layers. Using an Sigmoid function at the end of the model it transforms the linear tansformation into a probability (0,1) with threshold = 0.5 for detection of the event posibility for every timestamp of the timeseries.

# Dataset

The model is trained and evaluated using seismic waveform data sourced from the Northern California Earthquake Data Center (NCEDC), which provides high-quality digital records of central and northern California seismicity [1].

The empirical foundation of this project is based on the quakeflow_nc dataset, curated and distributed by the AI4EPS (Artificial Intelligence for Earthquake and Planet Sciences) research group processed in HDF5 format [2].

From quakeflow_nc was used:
1. 3-channel seismological waveform
2. phase_index
3. phase_type



    The dataset does not contain coda waves so we simulate the using a coda multiplier = 1.5

Seismic Event Duration = (Ts - Tp) * Coda_Multiplier where s and p represents the time that p and s waves arive







# References

[1]‘NCEDC: Northern California Earthquake Data Center’. Accessed: May 24, 2026. [Online]. Available: https://ncedc.org/ 

[2] AI4EPS, ‘quakeflow_nc’. Hugging Face. doi: 10.57967/HF/0716. 

# How to Run?

1. py download.py
2. py merge.py
3. py main.py
