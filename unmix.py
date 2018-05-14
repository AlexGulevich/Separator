
import numpy as np
from scipy.io import wavfile
from kica import kica
import argparse
import os
import math
import matplotlib.pyplot as plt

def check_not_exist(name):
    if not os.path.exists(name):
        print ('{0} is not available'.format(name))
        return -1
    else:
        return 0



def mserr(x, y):

    # center data
    x = x - x.mean()
    y = y - y.mean()

    # lead to unit variance
    x = x / np.std(x)  # todo: what if zero variance?
    y = y / np.std(y)

    # MSE itself
    return (( x - y )**2).mean()
    

def minMSE(zz, data):

    # unmixed data can be inverted
    mse1 = mserr(  zz[0, :], data)
    mse2 = mserr( -zz[0, :], data)
    mse3 = mserr(  zz[1, :], data)
    mse4 = mserr( -zz[1, :], data)

    # also unmixed data can be swapped every frame
    mses = [mse1, mse2, mse3, mse4]
    mse = min(mses)
    numb = np.argmin(mses)

    if   numb == 0:
        zz1 =  zz[0, :]
    elif numb == 1:
        zz1 = -zz[0, :]
    elif numb == 2:
        zz1 =  zz[1, :]
    elif numb == 3:
        zz1 = -zz[1, :]

    # mse and choosed unmixed channel
    return mse, zz1



def split(name1, name2, uxname1, uxname2):
    
    if check_not_exist(name1) or check_not_exist(name2):    
        return
    
    # Get the recorded signals.
    freq1, data1 = wavfile.read(name1)
    freq2, data2 = wavfile.read(name2)

   
    if freq1 != freq2:
        print('frequenies are not equal!. Cannot proceed')
        return

    data1 = data1[0:1000000]
    data2 = data2[0:1000000]
    
    # Assemble into a 2D array. Each row is a recording from a mic.
    x = np.vstack((data1, data2))


    frame_overlap = 0.25    # from 0.0 to 0.99
    frame_step =    800     # step:            frame_overlap * frame_step should be int
    frame_maxv = 100000     # frame max value 
    frame_minv =   4000     # frame min value: frame_minv    * frame_step should be int
    

    res  = []
    res2 = []
    time = []

    for frame in range(frame_minv, frame_maxv + 1, frame_step):

        mse1 = 0
        mse2 = 0
        zzz = [[],[]]

        for k in range(1, math.floor(( len(data1) - frame ) / ( frame * ( 1 - frame_overlap ) ) ) + 1 + 1):

            # indices for xx
            xn1 = round( ( k - 1) * frame * ( 1 - frame_overlap ) )
            xn2 = xn1 + frame

            # make frame
            xx = x[:, xn1:xn2]
            
            # Do kICA.
            w, yy = kica(xx)
    
            # Unmix the signal.
            zz = np.dot(w, yy)

            # indices for zz
            zn1 = round( frame * (     frame_overlap / 2 ) )
            zn2 = round( frame * ( 1 - frame_overlap / 2 ) )

            # indices for input data
            dn1 = xn1 + zn1
            dn2 = xn1 + zn2

            # mse and choose unmixed data
            mse1_frame, zz1 = minMSE(zz[:, zn1:zn2], data1[dn1:dn2])
            mse2_frame, zz2 = minMSE(zz[:, zn1:zn2], data2[dn1:dn2])
            
            #frame MSE
            mse1 = mse1 + mse1_frame
            mse2 = mse2 + mse2_frame

            # unmixed data
            zzz = np.append(zzz, np.vstack((zz1, zz2)), 1)

        #frame MSE
        res .append(math.sqrt( (mse1 + mse2) / 2 ))

        # global MSE
        mse1 = mserr( zzz[0, :], data1[zn1 : zn1 + len(zzz[0, :])])
        mse2 = mserr( zzz[1, :], data2[zn1 : zn1 + len(zzz[1, :])])
        res2.append(math.sqrt( (mse1 + mse2) / 2 ))

        # X axis
        time.append(frame / freq1);

    #plots
    plt.subplot(2,1,1)
    plt.plot(time, res )
    plt.ylabel("Frame MSE")

    plt.subplot(2,1,2)
    plt.plot(time, res2)

    plt.xlabel("Frame size in sec")
    plt.ylabel("Global MSE")
    
    plt.show()
    
    # Make new wav files containing the unmixed signals.
    wavfile.write(uxname1, freq1, 1000*zzz[0].astype(np.int16))
    wavfile.write(uxname2, freq1, 1000*zzz[1].astype(np.int16))
    

def merge (name1, name2, xname1, xname2, mix = np.array([1,0,0,1])):
    if check_not_exist(name1) or check_not_exist(name2):    
        return
     # Get the recorded signals.
    freq1, data1 = wavfile.read(name1)
    freq2, data2 = wavfile.read(name2)
    
    if freq1 != freq2:
        print('frequenies are not equal!. Cannot proceed')
        return
    if data1.dtype != data2.dtype:
        print ('types are not equal.')
        return
    gen_type = data1.dtype
    if np.float32 == gen_type:        
        (min_val, max_val) = (-1.0, 1.0)
    elif np.uint8 == gen_type:
        (min_val, max_val) = (0, 255)
    else:
        (min_val, max_val) = (np.iinfo(gen_type).min, np.iinfo(gen_type).max)
        
    min_len = np.min((data1.shape[0], data2.shape[0]))
    
    x = np.vstack((data1[:min_len], data2[:min_len]))
    
    mix = mix.reshape([2,2])
    mix_out = np.dot(mix, x)
    mix_out = np.clip(mix_out, min_val, max_val)
    
    # Make new wav files containing the unmixed signals.
    wavfile.write(xname1, freq1, mix_out[0].astype(gen_type))
    wavfile.write(xname2, freq1, mix_out[1].astype(gen_type))
    
    
    

def arguments():
    parser=argparse.ArgumentParser(description='split or mix')
    parser.add_argument('--type', required=True, type=str, help='either split or mix')
    parser.add_argument('--mix_matrix', nargs=4, default=[1,0,0,1], required=False)
    parser.add_argument('--in1', required=True)
    parser.add_argument('--in2', required=True)
    parser.add_argument('--out1', required=True)
    parser.add_argument('--out2', required=True)
    return parser
    

if __name__=='__main__':
    #parser = arguments()
    #args = parser.parse_args()
    #if 'split' == args.type:        
    #    split(args.in1, args.in2, args.out1, args.out2)
    #elif 'mix' == args.type:
    #    merge(args.in1, args.in2, args.out1, args.out2, np.array(args.mix_matrix).astype(np.float32))

    merge("wav/Alice_8k_8bit.wav", "wav/George_8k_8bit.wav", "wav/GA1.wav", "wav/GA2.wav", np.array([0.7, 0.3, 0.3, 0.7]).astype(np.float32))
    #merge("wav/OSR_fr_000_0041_8k.wav", "wav/OSR_us_000_0010_8k.wav", "wav/GA1.wav", "wav/GA2.wav", np.array([0.7, 0.3, 0.3, 0.7]).astype(np.float32))

    split("wav/GA1.wav", "wav/GA2.wav", "wav/ouuu1.wav", "wav/ouuu2.wav")
    
    print('operation is completed')
