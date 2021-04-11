# CNN4CE
Source codes of the article "Deep CNN-Based Channel Estimation for mmWave Massive MIMO Systems" in IEEE JSTSP

This folder contains codes for channel data generation (.m) and channel estimation using CNN (.py). 

Channel data generation:
1. Use channel_statistic_generation.m to generate multiple channel statistics.
2. Use channel_data_generation.m to generate multiple channel realizations under each channel statistic.

Channel estimation:
1. Use CNN_2bands_UMi_training.py to train the CNN and save model.
2. Use CNN_2bands_UMi_further_training.py to further train the CNN based on the saved model.
3. Use CNN_2bands_UMi_testing.py to test the performance of the trained CNN.
