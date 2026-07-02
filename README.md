# Earthquake Detection
Seismic Phase Picking 

This repository contains the implementation of a deep learning framework designed for automated Earthquake Event Detection in multivariate, 3-channel seismological time-series data. The project explores the transition from traditional anomaly detection to supervised event boundary delineation under real-world, noisy conditions and hardware constraints.

Model Architecture

1D-CNN Layers: Act as adaptive frequency filters to extract spatial features and local spectral characteristics from raw waveforms. First Filter -> Wide filter for the context. Second Filter -> Tight Filter for details

Bidirectional LSTM (BiLSTM): Captures temporal dependencies by processing the time series in both forward and backward directions, ensuring precise identification of event onset and offset boundaries (including the coda wave phase).

Linear Layer: The features from LSTM pass from an Linear Layers. Using an Sigmoid function at the end of the model it transforms the linear tansformation into a probability (0,1) with threshold = 0.5 for detection of the event posibility for every timestamp of the timeseries.

How to run?

py download.py 
py merge.py
py main.py 
