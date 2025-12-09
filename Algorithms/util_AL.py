"""
Copyright (C) 2019-2023 Luis Howell & Bernd Porr
GPL GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
"""

import numpy as np
import scipy.signal as signal
import pandas as pd

class Detectors:
    """ECG heartbeat detection algorithms
    General useage instructions:
    r_peaks = detectors.the_detector(ecg_in_samples)
    The argument ecg_in_samples is a single channel ECG in volt
    at the given sample rate.
    """

    def __init__(self, sampling_frequency = False):
        """
        The constructor takes the sampling rate in Hz of the ECG data.
        The constructor can be called without speciying a sampling rate to
        just access the detector_list, however, detection won't
        be possible.
        """

        ## Sampling rate
        self.fs = sampling_frequency

        ## This is set to a positive value for benchmarking
        self.engzee_fake_delay = 0

        ## 2D Array of the different detectors: [[description,detector]]
        # self.detector_list = [
        #     ["Elgendi et al (Two average)",self.two_average_detector],
        #     ["Matched filter",self.matched_filter_detector],
        #     ["Kalidas & Tamil (Wavelet transform)",self.swt_detector],
        #     ["Engzee",self.engzee_detector],
        #     ["Christov",self.christov_detector],
        #     ["Hamilton",self.hamilton_detector],
        #     ["Pan Tompkins",self.pan_tompkins_detector],
        #     ["WQRS",self.wqrs_detector]
        # ]
        self.detector_list = [["Pan Tompkins",self.pan_tompkins_detector]]

    def pan_tompkins_detector(self, unfiltered_ecg, MWA_name='cumulative'):
        """
        Jiapu Pan and Willis J. Tompkins.
        A Real-Time QRS Detection Algorithm. 
        In: IEEE Transactions on Biomedical Engineering 
        BME-32.3 (1985), pp. 230–236.
        """
        
        maxQRSduration = 0.150 #sec
        f1 = 5/self.fs
        f2 = 15/self.fs

        b, a = signal.butter(1, [f1*2, f2*2], btype='bandpass')

        filtered_ecg = signal.lfilter(b, a, unfiltered_ecg)        

        diff = np.diff(filtered_ecg) 

        squared = diff*diff

        N = int(maxQRSduration*self.fs)
        mwa = MWA_from_name(MWA_name)(squared, N)
        mwa[:int(maxQRSduration*self.fs*2)] = 0

        mwa_peaks = panPeakDetect(mwa, self.fs)

        return mwa_peaks



def panPeakDetect(detection, fs):    

    min_distance = int(0.25*fs)

    signal_peaks = [0]
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    peaks = []

    for i in range(1,len(detection)-1):
        if detection[i-1]<detection[i] and detection[i+1]<detection[i]:
            peak = i
            peaks.append(i)

            if detection[peak]>threshold_I1 and (peak-signal_peaks[-1])>0.3*fs:
                    
                signal_peaks.append(peak)
                indexes.append(index)
                SPKI = 0.125*detection[signal_peaks[-1]] + 0.875*SPKI
                if RR_missed!=0:
                    if signal_peaks[-1]-signal_peaks[-2]>RR_missed:
                        missed_section_peaks = peaks[indexes[-2]+1:indexes[-1]]
                        missed_section_peaks2 = []
                        for missed_peak in missed_section_peaks:
                            if missed_peak-signal_peaks[-2]>min_distance and signal_peaks[-1]-missed_peak>min_distance and detection[missed_peak]>threshold_I2:
                                missed_section_peaks2.append(missed_peak)

                        if len(missed_section_peaks2)>0:
                            signal_missed = [detection[i] for i in missed_section_peaks2]
                            index_max = np.argmax(signal_missed)
                            missed_peak = missed_section_peaks2[index_max]
                            missed_peaks.append(missed_peak)
                            signal_peaks.append(signal_peaks[-1])
                            signal_peaks[-2] = missed_peak   

            else:
                noise_peaks.append(peak)
                NPKI = 0.125*detection[noise_peaks[-1]] + 0.875*NPKI

            threshold_I1 = NPKI + 0.25*(SPKI-NPKI)
            threshold_I2 = 0.5*threshold_I1

            if len(signal_peaks)>8:
                RR = np.diff(signal_peaks[-9:])
                RR_ave = int(np.mean(RR))
                RR_missed = int(1.66*RR_ave)

            index = index+1      
    
    signal_peaks.pop(0)

    return signal_peaks

def MWA_from_name(function_name):
    if function_name == "cumulative":
        return MWA_cumulative
    elif function_name == "convolve":
        return MWA_convolve
    elif function_name == "original":
        return MWA_original
    else: 
        raise RuntimeError('invalid moving average function!')

#Fast implementation of moving window average with numpy's cumsum function 
def MWA_cumulative(input_array, window_size):
    
    ret = np.cumsum(input_array, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    
    for i in range(1,window_size):
        ret[i-1] = ret[i-1] / i
    ret[window_size - 1:]  = ret[window_size - 1:] / window_size
    
    return ret

#Original Function 
def MWA_original(input_array, window_size):

    mwa = np.zeros(len(input_array))
    mwa[0] = input_array[0]
    
    for i in range(2,len(input_array)+1):
        if i < window_size:
            section = input_array[0:i]
        else:
            section = input_array[i-window_size:i]        
        
        mwa[i-1] = np.mean(section)

    return mwa

#Fast moving window average implemented with 1D convolution 
def MWA_convolve(input_array, window_size):
    
    ret = np.pad(input_array, (window_size-1,0), 'constant', constant_values=(0,0))
    ret = np.convolve(ret,np.ones(window_size),'valid')
    
    for i in range(1,window_size):
        ret[i-1] = ret[i-1] / i
    ret[window_size-1:] = ret[window_size-1:] / window_size
    
    return ret

def compute_rr(signal, track):

    signal_clean = signal[signal[track].notna()]


    ecg_signal = np.array(signal_clean[track], dtype=np.float64)
    timestamps = np.array(signal_clean["Time"], dtype=np.float64)

    # Generate times vector
    times = np.arange(len(ecg_signal)) / 500

    # Pan-Tompkins Detector
    detectors = Detectors(500)
    r_peaks_ind = detectors.pan_tompkins_detector(ecg_signal)

    # Get the peaks timestamps indexes
    timestamps_indexes = timestamps[r_peaks_ind]

    # Compute the R-R intervals (seconds)
    r_peaks_times = times[r_peaks_ind]
    
    #RR_Df : Timestamp_ini | Timestamp_fin | rr
    return pd.DataFrame({'Time_ini_ms': np.delete(timestamps_indexes, len(timestamps_indexes)-1 ), 'Time_fin_ms': np.delete(timestamps_indexes, 0), 'rr': np.diff(r_peaks_times)})


def check_availability(tracks): #Function that checks whichs algorithms can be computed.
    possible_list = []

    if ('Intellivue/ECG_HR' in tracks or 'Intellivue/ABP_HR' in tracks or 'Intellivue/HR' in tracks) and ('Intellivue/ABP_SYS' in tracks or 'Intellivue/BP_SYS' in tracks or 'Intellivue/NIBP_SYS' in tracks):
        possible_list.append('Shock Index')
    if 'Intellivue/PPLAT_CMH2O' in tracks and 'Intellivue/PEEP_CMH2O' in tracks:
        possible_list.append('Driving Pressure')
    if 'Intellivue/TV_EXP' in tracks and 'Intellivue/PIP_CMH2O' in tracks and 'Intellivue/PEEP_CMH2O' in tracks:
        possible_list.append('Dynamic Compliance')
    if 'Intellivue/PLETH_SAT_O2' in tracks and 'Intellivue/FiO2' in tracks:
        possible_list.append('ROX Index')
    if ('Intellivue/BT_CORE' in tracks or 'Intellivue/BT_BLD' in tracks) and ('Intellivue/BT_SKIN' in tracks or 'Intellivue/TEMP' in tracks):
        possible_list.append('Temp Comparison')
    #MostCare
    if 'Intellivue/VOL_BLD_STROKE' in tracks and ('Intellivue/ECG_HR' in tracks or 'Intellivue/ABP_HR' in tracks or 'Intellivue/HR' in tracks):
        possible_list.append('Cardiac Output')
    if ('Intellivue/ABP_MEAN' in tracks or 'Intellivue/BP_MEAN' in tracks or 'Intellivue/NIBP_MEAN' in tracks) and 'Intellivue/CVP_MEAN' in tracks and 'Cardiac Output' in possible_list:
        possible_list.append('Systemic Vascular Resistance')
    if ('Intellivue/ABP_MEAN' in tracks or 'Intellivue/BP_MEAN' in tracks or 'Intellivue/NIBP_MEAN' in tracks) and 'Cardiac Output' in possible_list:
        possible_list.append('Cardiac Power Output')
    if ('Intellivue/ABP_SYS' in tracks or 'Intellivue/BP_SYS' in tracks or 'Intellivue/NIBP_SYS' in tracks) and 'Intellivue/VOL_BLD_STROKE' in tracks:
        possible_list.append('Effective Arterial Elastance')

    #Variables autonomicas
    if 'Intellivue/ECG_I' in tracks or 'Intellivue/ECG_II' in tracks or 'Intellivue/ECG_III' in tracks or 'Intellivue/ECG_V' in tracks:
        possible_list.append('Heart Rate Variability') #NO SE PODRÁ MOSTRAR POR PANTALLA, SOLO INVESTIGAÇAO
    if 'Heart Rate Variability' in possible_list and 'Intellivue/ABP' in tracks:
        possible_list.append('Blood Pressure Variability') #NO SE PODRÁ MOSTRAR POR PANTALLA, SOLO INVESTIGAÇAO
    if 'Heart Rate Variability' in possible_list and 'Intellivue/ART' in tracks:
        possible_list.append('BRS') #NO SE PODRÁ MOSTRAR POR PANTALLA, SOLO INVESTIGAÇAO
    if 'Heart Rate Variability' in possible_list and 'Intellivue/CO2' in tracks or 'Intellivue/RESP' in tracks:
        possible_list.append('RSA') #NO SE PODRÁ MOSTRAR POR PANTALLA, SOLO INVESTIGAÇAO
        
    #Models
    if 'Intellivue/ICP' in tracks:
        possible_list.append('ICP Model')
    if 'Intellivue/PLETH' in tracks and 'Intellivue/ART' in tracks and 'Intellivue/ABP' in tracks:
        possible_list.append('ABP Model')

    return possible_list
