# CNN4CE
Source codes of the article: P. Dong, H. Zhang, G. Y. Li, I. S. Gaspar and N. NaderiAlizadeh, “Deep CNN-based channel estimation for mmWave massive MIMO systems,” IEEE J. Sel. Topics Signal Process., vol. 13, no. 5, pp. 989–1000, Sep. 2019. Please cite this paper when using the codes.

This folder contains codes for channel data generation executed in MATLAB and codes for channel estimation executed in Python. 

Channel data generation:
1. Use channel_statistic_generation.m to generate multiple channel statistics.
2. Use MIMO_channel_3GPP_multi_fre.m to generate channel data for SF-CNN.
3. Use MIMO_channel_3GPP_multi_fre_time.m to generate channel data for SFT-CNN and SPR-CNN.

Hints: For Release R2018b and beyond, use 5G Toolbox and "nrCDLChannel" instead. For more information, please refer to https://www.mathworks.com/matlabcentral/fileexchange/61585-lte-system-toolbox-5g-library and https://www.mathworks.com/products/5g.html

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
