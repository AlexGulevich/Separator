
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
    # cause unmixed data not scaled equal to original
    dev = np.std(x)
    if dev > 0:
        x = x / dev

    dev = np.std(y)
    if dev > 0:
        y = y / dev

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


def check_split_params(frame_overlap, frame_step, frame_minv):

    if frame_overlap < 0 or frame_overlap > 0.99:
        print('frame_overlap should be 0 < frame_overlap < 0.99')
        return -1

    if abs(round(frame_overlap * frame_step ) - frame_overlap * frame_step) > 0:
        print('frame_overlap * frame_step should be int')
        return -1

    if abs(round(frame_minv * frame_step ) - frame_minv * frame_step) > 0:
        print('frame_minv * frame_step should be int')
        return -1

    return 0
    

def split(ref1, ref2, mix_out, freq, uxname1, uxname2, frame_overlap, frame_step, frame_maxv, frame_minv):
    
    if check_split_params(frame_overlap, frame_step, frame_minv):
        return

    data1 = mix_out[0, :]
    data2 = mix_out[1, :]
    
    # Assemble into a 2D array. Each row is a recording from a mic.
    x = np.vstack((data1, data2))

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
            mse1_frame, zz1 = minMSE(zz[:, zn1:zn2], ref1[dn1:dn2])
            mse2_frame, zz2 = minMSE(zz[:, zn1:zn2], ref2[dn1:dn2])
            
            # frame MSE
            mse1 = mse1 + mse1_frame
            mse2 = mse2 + mse2_frame

            # unmixed data
            zzz = np.append(zzz, np.vstack((zz1, zz2)), 1)

        # frame MSE
        res .append(math.sqrt( (mse1 + mse2) / 2 ))

        # global MSE
        mse1 = mserr( zzz[0, :], data1[zn1 : zn1 + len(zzz[0, :])])
        mse2 = mserr( zzz[1, :], data2[zn1 : zn1 + len(zzz[1, :])])
        res2.append(math.sqrt( (mse1 + mse2) / 2 ))

        # X axis
        time.append(frame / freq);

    # plots
    plt.subplot(2,1,1)
    plt.plot(time, res )
    plt.ylabel("Frame MSE")

    plt.subplot(2,1,2)
    plt.plot(time, res2)

    plt.xlabel("Frame size in sec")
    plt.ylabel("Global MSE")
    
    plt.show()
    
    # Make new wav files containing the unmixed signals.
    wavfile.write(uxname1, freq, 1000*zzz[0].astype(np.int16))
    wavfile.write(uxname2, freq, 1000*zzz[1].astype(np.int16))
    

def merge (name1, name2, xname1, xname2, mix = np.array([1,0,0,1])):
    if check_not_exist(name1) or check_not_exist(name2):    
        return

    x = [[],[]]
    data1all = []
    data2all = []
    dir2 = os.listdir(name2)
    dir2Count = 0;
    firstFile = True
    
    for file in os.listdir(name1):

        if file.endswith(".wav"):
            # Get the recorded signal.
            freq1, data1 = wavfile.read(os.path.join(name1, file))

            if firstFile:
                freq = freq1
                dtype = data1.dtype
                firstFile = False

            while dir2Count < len(dir2):
                file = dir2[dir2Count]
                if file.endswith(".wav"):
                    break
                else:
                    dir2Count += 1
                    
            if dir2Count >= len(dir2):
                break

        # Get the recorded signal.
        freq2, data2 = wavfile.read(os.path.join(name2, file))
        dir2Count += 1
    
        if freq1 != freq2 and freq1 != freq:
            print('frequenies are not equal!. Cannot proceed')
            return
        if data1.dtype != data2.dtype and data1.dtype != dtype:
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

        x1 = np.vstack((data1[:min_len], data2[:min_len]))

        x = np.append(x, x1, 1)

        data1all = np.append(data1all, data1)
        data2all = np.append(data2all, data2)
    
    mix = mix.reshape([2,2])
    mix_out = np.dot(mix, x)
    mix_out = np.clip(mix_out, min_val, max_val)
    
    # Make new wav files containing the unmixed signals.
    wavfile.write(xname1, freq1, mix_out[0].astype(gen_type))
    wavfile.write(xname2, freq1, mix_out[1].astype(gen_type))

    return data1all, data2all, mix_out, freq1
    
    

def arguments():
    parser=argparse.ArgumentParser(description='split or mix')
    parser.add_argument('--type', required=True, type=str, help='either split or mix')
    parser.add_argument('--mix_matrix', nargs=4, default=[0.7, 0.3, 0.3, 0.7], required = False)
    parser.add_argument('--in1',  required = True)
    parser.add_argument('--in2',  required = True)
    parser.add_argument('--out1', required = True)
    parser.add_argument('--out2', required = True)
    parser.add_argument('--frame_overlap', default = 0, required = False)
    parser.add_argument('--frame_step',    default = 400,   required = False)
    parser.add_argument('--frame_maxv',    default = 50000, required = False)
    parser.add_argument('--frame_minv',    default =  3200,  required = False)
    return parser

   

if __name__=='__main__':
    parser = arguments()
    args = parser.parse_args()
    if 'split' == args.type:
        data1, data2, mix_out, freq = merge(args.in1, args.in2, args.out1, args.out2, np.array(args.mix_matrix).astype(np.float32))
        split(data1, data2, mix_out, freq, args.out1, args.out2, float(args.frame_overlap), int(args.frame_step), int(args.frame_maxv), int(args.frame_minv))
        
    elif 'mix' == args.type:
        merge(args.in1, args.in2, args.out1, args.out2, np.array(args.mix_matrix).astype(np.float32))

    print('operation is completed')
