# CNN4CE
Source codes of the article "Deep CNN-Based Channel Estimation for mmWave Massive MIMO Systems" in IEEE JSTSP

This folder contains codes for channel data generation (.m) and channel estimation using CNN (.py). 

Channel data generation:
1. Use channel_statistic_generation.m to generate multiple channel statistics.
2. Use channel_data_generation.m to generate multiple channel realizations under each channel statistic.

Channel estimation:
1. SF-CNN

       (1) Use SF_CNN_2fre_train.py to train the CNN and save model.

       (2) Use SF_CNN_2fre_train_further.py to further train the CNN based on the saved model.

       (3) Use SF_CNN_2fre_test.py to test the performance of the trained CNN.

2. SFT-CNN

       (1) Use SFT_CNN_2fre2time_train.py to train the CNN and save model.

       (2) Use SFT_CNN_2fre2time_train_further.py to further train the CNN based on the saved model.

       (3) Use SFT_CNN_2fre2time_test.py to test the performance of the trained CNN.
       
3. SPR-CNN

       (1) Use SPR_CNN1_train.py to train the SPR-CNN1 and save model.

       (2) Use SPR_CNN1_train_further.py to further train the SPR-CNN1 based on the saved model.

       (3) Use SPR_CNN1_test.py to test the performance of the trained SPR-CNN1.
       
       (4) SPR-CNN2, 3, and 4 can be trained and tested similarly.
