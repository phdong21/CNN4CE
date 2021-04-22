from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPool2D, normalization
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from numpy import *
import numpy as np
import numpy.linalg as LA
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #allow growth
session = tf.Session(config=config)
import scipy.io as sio

Nt=32
Nr=16
SNR=10.0**(20/10.0) # transmit power
# DFT matrix
def DFT_matrix(N):
    m, n = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1j / N )
    D = np.power( omega, m * n )
    return D

def sub_dftmtx(A, N):
    D=A[:,0:N]
    return D

F_DFT=DFT_matrix(Nt)/np.sqrt(Nt)
W_DFT=DFT_matrix(Nr)/np.sqrt(Nr)

Nt_beam=32
F_RF=F_DFT[:,0:Nt_beam]
F=F_RF
F_conj=np.conjugate(F)
F_conjtransp=np.transpose(F_conj)
FFH=np.dot(F,F_conjtransp)
Nr_beam=16
W_RF=W_DFT[:,0:Nr_beam]
W=W_RF
W_conj=np.conjugate(W)
W_conjtransp=np.transpose(W_conj)
WWH=np.dot(W,W_conjtransp)

Nt_beam1=16
F_RF1=F_DFT[:,0:Nt_beam1]
F1=F_RF1
F_conj1=np.conjugate(F1)
F_conjtransp1=np.transpose(F_conj1)
FFH1=np.dot(F1,F_conjtransp1)
Nr_beam1=4
W_RF1=W_DFT[:,0:Nr_beam1]
W1=W_RF1
W_conj1=np.conjugate(W1)
W_conjtransp1=np.transpose(W_conj1)
WWH1=np.dot(W1,W_conjtransp1)

Nt_beam2=16
F_RF2=F_DFT[:,0:Nt_beam2]
F2=F_RF2
F_conj2=np.conjugate(F2)
F_conjtransp2=np.transpose(F_conj2)
FFH2=np.dot(F1,F_conjtransp2)
Nr_beam2=4
W_RF2=W_DFT[:,0:Nr_beam2]
W2=W_RF2
W_conj2=np.conjugate(W2)
W_conjtransp2=np.transpose(W_conj2)
WWH2=np.dot(W2,W_conjtransp2)

Nt_beam3=16
F_RF3=F_DFT[:,0:Nt_beam3]
F3=F_RF3
F_conj3=np.conjugate(F3)
F_conjtransp3=np.transpose(F_conj3)
FFH3=np.dot(F3,F_conjtransp3)
Nr_beam3=4
W_RF3=W_DFT[:,0:Nr_beam3]
W3=W_RF3
W_conj3=np.conjugate(W3)
W_conjtransp3=np.transpose(W_conj3)
WWH3=np.dot(W3,W_conjtransp3)

scale=2
fre=2
time_steps=4

############## testing set generation ##################
data_num_test=500
data_num_file=500
H_test=zeros((data_num_test,Nr,Nt,2*fre), dtype=float)
H_test_noisy=zeros((data_num_test,Nr,Nt,2*fre*time_steps), dtype=float)
filedir = os.listdir('D:\\2fre4time_data')
n=0
SNRr=0
SNR_factor=5.9
for filename in filedir:
    newname = os.path.join('D:\\2fre4time_data', filename)
    data = sio.loadmat(newname)
    channel = data['ChannelData_fre']
    for i in range(data_num_file):
        for j in range(fre):
            a=channel[:,:,j,i]
            for t in range(time_steps):
                a1=a[:,t*Nr:t*Nr+Nr]
                H=np.transpose(a1)
                H_re=np.real(H)
                H_im = np.imag(H)
                if t==3:
                    H_test[n * data_num_file + i, :, :, 2 * j] = H_re / scale
                    H_test[n * data_num_file + i, :, :, 2 * j + 1] = H_im / scale
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam))
                NFH = np.dot(N, F_conjtransp)
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam1)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam1))
                NFH1 = np.dot(N, F_conjtransp1)
                N1 = np.dot(WWH1, NFH1)
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam2)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam2))
                NFH2 = np.dot(N, F_conjtransp2)
                N2 = np.dot(WWH2, NFH2)
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam3)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam3))
                NFH3 = np.dot(N, F_conjtransp3)
                N3 = np.dot(WWH3, NFH3)
                if t == 0:
                    Y = H + 1.0 / np.sqrt(SNR_factor * SNR) * NFH
                    SNRr = SNRr + SNR_factor * SNR * (LA.norm(H)) ** 2 / (LA.norm(NFH)) ** 2
                    Y_re = np.real(Y)
                    Y_im = np.imag(Y)
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t] = Y_re / scale
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t + 1] = Y_im / scale
                if t == 1:
                    HF1 = np.dot(H, FFH1)
                    H1 = np.dot(WWH1, HF1)
                    Y = H1 + 1.0 / np.sqrt(SNR_factor * SNR) * N1
                    SNRr = SNRr + SNR_factor * SNR * (LA.norm(H1)) ** 2 / (LA.norm(N1)) ** 2
                    Y_re = np.real(Y)
                    Y_im = np.imag(Y)
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t] = Y_re / scale
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t + 1] = Y_im / scale
                if t == 2:
                    HF2 = np.dot(H, FFH2)
                    H2 = np.dot(WWH2, HF2)
                    Y = H2 + 1.0 / np.sqrt(SNR_factor * SNR) * N2
                    SNRr = SNRr + SNR_factor * SNR * (LA.norm(H2)) ** 2 / (LA.norm(N2)) ** 2
                    Y_re = np.real(Y)
                    Y_im = np.imag(Y)
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t] = Y_re / scale
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t + 1] = Y_im / scale
                if t == 3:
                    HF3 = np.dot(H, FFH3)
                    H3 = np.dot(WWH3, HF3)
                    Y = H3 + 1.0 / np.sqrt(SNR_factor * SNR) * N3
                    SNRr = SNRr + SNR_factor * SNR * (LA.norm(H3)) ** 2 / (LA.norm(N3)) ** 2
                    Y_re = np.real(Y)
                    Y_im = np.imag(Y)
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t] = Y_re / scale
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t + 1] = Y_im / scale
    n = n + 1
print(n)
print(SNRr/(data_num_test*fre*time_steps))
print(H_test.shape,H_test_noisy.shape)
index3 = np.where(abs(H_test) > 1)
row_num = np.unique(index3[0])
H_test = np.delete(H_test, row_num, axis=0)
H_test_noisy = np.delete(H_test_noisy, row_num, axis=0)
print(len(row_num))
print(H_test.shape, H_test_noisy.shape)
print(((H_test)**2).mean())

# load model
CNN = load_model('2fre4time_SNR20_time4_16_4_600ep.hdf5')

decoded_channel = CNN.predict(H_test_noisy)
nmse2=zeros((data_num_test-len(row_num),1), dtype=float)
for n in range(data_num_test-len(row_num)):
    MSE=((H_test[n,:,:,:]-decoded_channel[n,:,:,:])**2).sum()
    norm_real=((H_test[n,:,:,:])**2).sum()
    nmse2[n]=MSE/norm_real
print(nmse2.sum()/(data_num_test-len(row_num)))# calculate NMSE after training stage (testing performance)