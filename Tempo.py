import numpy as np
import librosa
import pywt
import pywt.data
import math

from sklearn import mixture
import scipy.stats as stats
'''
Cause I can't find the ISMIR 2004 dataset.
Test on Mirex competition
https://www.music-ir.org/mirex/wiki/2018:MIREX2018_Results
https://www.music-ir.org/mirex/wiki/2018:Audio_Tempo_Estimation#Description

Download GiantSteps tempo from here:
https://github.com/GiantSteps/giantsteps-tempo-dataset

Dataset:
http://www.audiocontentanalysis.org/data-sets/
'''

def getWeights(shifts:np.ndarray):
    """
    shifts : ndarray with shape (1, 10)
    Return the number of beat intervals that falls within [tar_beat_interval-IOI_deviation, tar_beat_interval+IOI_deviation]
    """
    shifts = shifts.squeeze()
    assert (shifts.shape[0]==10)
    
    tar_beat_interval = shifts[0]
    IOI_deviation = shifts[-1]
    shifts = shifts[1:-1]
    
    indices = np.where(abs(shifts - tar_beat_interval) <= IOI_deviation)[0]
    #return 1 # No weights
    return indices.shape[0]+1

def estimateTempo(y:np.ndarray, sr:float, verbose=True, IOI_fix=-1.0, IOI_factor=-1.0) -> float:
    occ_g_all = []
    n_comp = 8

    # 5.3 dwt for cA1~cA4
    cDs = []
    cA_last = y
    for i in range(4):
        cA, cD = pywt.dwt(cA_last, 'db5')
        cDs.append(cD)
        cA_last = cA

    # 5.4 Peak Detection for each scale
    for i, cD in enumerate(cDs):
        # 5.4.1 Full wave rectification
        cD = np.abs(cD)
        
        # 5.4.2 Moving window
        # Assume the max tempo is 240bpm -> 0.25sec -> window size: 0.25sec
        # Step size: 1/20 of the window length
        # Sample rate in this scale
        c_sr = int(sr * (1/2)**(i+1))
        window_len = int(math.floor(0.25*c_sr))
        step_len = int(math.floor(window_len/20))
        if verbose:
            print (f"Current SR:{c_sr}, Window size:{window_len}, Step size:{step_len}")
        
        peaks = np.zeros_like(cD, dtype=np.int)
        j = 0
        while j + window_len < cD.shape[0]:
            p_cD = cD[j:j+window_len]
            indices = np.argwhere(p_cD==np.amax(p_cD)).flatten().tolist()
            indices = [index + j for index in indices]
            peaks[indices] += 1
            
            j = j + step_len
            
        # Remove peaks less than 18 times
        peaks = np.where(peaks<18, 0,  peaks)
        
        # 5.5 Estimate beat intervals
        peak_indices = np.where(peaks>0)[0]
        beat_intervals = (peak_indices - np.roll(peak_indices, 1))[1:] # The distance(in samples) betwen this peak and the last peak 
        beat_intervals = beat_intervals / c_sr # Convert samples to sec
        bpms = 60/beat_intervals
        
        # 5.7.1 IOI deviation y=320.67*x**(-0.3388), see Chapter 3
        IOI_deviations = np.zeros_like(bpms)
        if IOI_fix != -1.0:
            IOI_deviations = np.ones_like(bpms)*IOI_fix/1000 # sec
        elif IOI_factor != -1.0:
            IOI_deviations = np.ones_like(bpms)*beat_intervals*IOI_factor # sec
        else:
            IOI_deviations = 320.67*np.power(bpms, -0.3388)/1000 # sec
        bpms = np.round(bpms).astype(np.int)
        
        # 5.7.1 Weights
        # beat_intervals_2d with shape (beat_intervals, 10) with unit=sec
        #  [:, 0] = beat_intervals
        #  [:, 1~4] = beat_intervals L shift 4~1
        #  [:, 5~8] = beat_intervals R shift 1~4
        #  [:, 9] = IOI_deviations
        beat_intervals_2d = np.transpose(np.array([np.roll(beat_intervals, i) for i in [0, -4, -3, -2, -1, 1, 2, 3, 4]]))
        beat_intervals_2d = np.concatenate((beat_intervals_2d, IOI_deviations.reshape(-1, 1)), axis=1)
        weights = np.apply_along_axis(getWeights, 1, beat_intervals_2d)
        
        # Silly way to expand weights
        occ_g = [] # Histogram in this scale
        for bpm, weight in zip(bpms, weights):
            while bpm > 240:
                bpm /= 2
            occ_g += [bpm]*weight
        occ_g_all += occ_g # Histogram in all scale
                
        # 5.7.2 Smooth the histogram
        #occ_g = np.array(occ_g).reshape(-1, 1)
        #g = mixture.GaussianMixture(n_components=n_comp,covariance_type='full')
        #g.fit(occ_g)
        
        #ws = g.weights_.ravel()
        #ms = g.means_.ravel()
        #cs = g.covariances_.ravel()
        
        
    # Made the final decision
    # 5.7.2 Smooth the histogram
    occ_g = occ_g_all
    occ_g = np.array(occ_g).reshape(-1, 1)
    g = mixture.GaussianMixture(n_components=n_comp,covariance_type='full')
    g.fit(occ_g)

    ws = g.weights_.ravel()
    ms = g.means_.ravel()
    cs = g.covariances_.ravel()

    inc_list = np.argsort(ws)
    pred_tempo1, pred_tempo2 = ms[inc_list[-1]], ms[inc_list[-2]]
    return pred_tempo1, ws[inc_list[-1]], pred_tempo2, ws[inc_list[-2]]
