from decimal import ROUND_HALF_UP, Decimal
from logging import getLogger
import math
import traceback
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy import stats
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.regression.linear_model import burg

logger = getLogger(__name__)


class Preprocess:
    def __init__(self, fs: int = 50) -> None:
        """
        Args:
            fs (int, default=50): Sampling frequency of sensor signals
        """
        self.fs = fs

    def apply_filter(
        self, signal: pd.DataFrame, filter: str = "median", window: int = 5
    ) -> pd.DataFrame:
        """A denosing filter is applied to remove noise in signals.
        Args:
            signal (pd.DataFrame): Raw signal
            filter (str, default='median'): Filter name is chosen from 'mean', 'median', or 'butterworth'
            window (int, default=5): Length of filter
        Returns:
            signal (pd.DataFrame): Filtered signal
        See Also:
            'butterworth' applies a 3rd order low-pass Butterworth filter with a corner frequency of 20 Hz.
        """
        if filter == "mean":
            signal = signal.rolling(window=window, center=True, min_periods=1).mean()
        elif filter == "median":
            signal = signal.rolling(window=window, center=True, min_periods=1).median()
        elif filter == "butterworth":
            fc = 20  # cutoff frequency
            w = fc / (self.fs / 2)  # Normalize the frequency
            b, a = butter(3, w, "low")  # 3rd order low-pass Butterworth filter
            signal = pd.DataFrame(filtfilt(b, a, signal, axis=0), columns=signal.columns)
        else:
            try:
                raise ValueError("Not defined filter. See Args.")
            except ValueError:
                logger.error(traceback.format_exc())

        return signal

    def normalize(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization
        Args:
            signal (pd.DataFrame): Raw signal
        Returns:
            signal (pd.DataFrame): Normalized signal
        """
        df_mean = signal.mean()
        df_std = signal.std()
        signal = (signal - df_mean) / df_std
        return signal


    def segment_signal(
        self,
        signal: pd.DataFrame,
        window_size: int,
        # overlap_rate: int = 0.5,
        overlap: int,
        res_type: str = "dataframe",
    ) -> List[pd.DataFrame]:
        """Sample sensor signals in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window).
        Args:
            signal (pandas.DataFrame): Raw signal
            window_size (int, default=128): Window size of sliding window to segment raw signals.
            overlap (int, default=10): Overlap of sliding window to segment raw signals.
            res_type (str, default='dataframe'): Type of return value; 'array' or 'dataframe'
        Returns:
            signal_seg (list of pandas.DataFrame): List of segmented sigmal.
        """
        signal_seg = []

        for start_idx in range(0, len(signal) + 1 - window_size, overlap):
            seg = signal.iloc[start_idx : start_idx + window_size].reset_index(drop=True)
            if res_type == "array":
                seg = seg.values
            signal_seg.append(seg)

        if res_type == "array":
            signal_seg = np.array(signal_seg)

        return signal_seg

    def separate_gravity(self, acc: pd.DataFrame, cal_attitude_angle=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separate acceleration signal into body and gravity acceleration signal.
        Another low pass Butterworth filter with a corner frequency of 0.3 Hz is applied.
        Args:
            acc (pd.DataFrame): Segmented acceleration signal
        Returns:
            acc_body (pd.DataFrame): Body acceleration signal
            acc_grav (pd.DataFrame): Gravity acceleration signal
        """
        fc = 0.3  # cutoff frequency
        w = fc / (self.fs / 2)  # Normalize the frequency
        b, a = butter(3, w, "low")  # 3rd order low pass Butterworth filter
        acc_grav = pd.DataFrame(
            filtfilt(b, a, acc, axis=0),
            index=acc.index, columns=acc.columns
        )  # Apply Butterworth filter

        # Substract gravity acceleration from acceleration sigal.
        acc_body = acc.sub(acc_grav)
        
        ####cal_attitude_angle####
        if cal_attitude_angle == True:
            filtered_acc_grav = acc_grav.values
            grav_angle = np.zeros_like(filtered_acc_grav)
            grav_angle[:,0] = np.arctan2(filtered_acc_grav[:,1],filtered_acc_grav[:,2])
            grav_angle[:,1] = np.arctan2(filtered_acc_grav[:,0],filtered_acc_grav[:,2])
            grav_angle[:,2] = np.arctan2(filtered_acc_grav[:,0],filtered_acc_grav[:,1])
            grav_angle = pd.DataFrame(
                grav_angle,
                index=acc.index, columns=acc.columns
            )
            return acc_body, grav_angle
        ####cal_attitude_angle####
        else:
            return acc_body, acc_grav

    def obtain_jerk_signal(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Derive signal to obtain Jerk signals
        Args:
            signal (pd.DataFrame)
        Returns:
            jerk_signal (pd.DataFrame):
        """
        jerk_signal = signal.diff(periods=1)  # Calculate difference
        jerk_signal.iloc[0] = jerk_signal.iloc[1]  # Fillna
        jerk_signal = jerk_signal / (1 / self.fs)  # Derive in time (1 / sampling frequency)
        return jerk_signal

    def obtain_magnitude(self, signal):
        """Calculate the magnitude of these three-dimensional signals using the Euclidean norm
        Args:
            signal (pandas.DataFrame): Three-dimensional signals
        Returns:
            res (pandas.DataFrame): Magnitude of three-dimensional signals
        """
        return pd.DataFrame(norm(signal, ord=2, axis=1))

    def obtain_spectrum(self, signal):
        """Obtain spectrum using Fast Fourier Transform (FFT).
        Args:
            signal (pandas.DataFrame): Time domain signals
        Returns:
            amp (pandas.DataFrame): Amplitude spectrum
            phase (pandas.DataFrame): Phase spectrum
        """
        N = len(signal)
        columns = signal.columns

        for col in columns:
            signal[col] = signal[col] * np.hamming(N)  # hamming window

        F = fft(signal, axis=0)  # Apply FFT
        F = F[: N // 2, :]  # Remove the overlapping part

        amp = np.abs(F)  # Obtain the amplitude spectrum
        amp = amp / N * 2
        amp[0] = amp[0] / 2
        amp = pd.DataFrame(amp, columns=columns)  # Convert array to DataFrame
        phase = np.angle(F)
        phase = pd.DataFrame(phase, columns=columns)  # Convert array to DataFrame

        return amp, phase

    def obtain_ecdf_percentile(self, signal, n_bins=10):
        """Obtain ECDF (empirical cumulative distribution function) percentile values.
        Args:
            signal (DataFrame): Time domain signals
            n_bins (int, default: 10): How many percentiles to use as a feature
        Returns:
            features (array): ECDF percentile values.
        """
        idx = np.linspace(0, signal.shape[0] - 1, n_bins)  # Take n_bins linspace percentile.
        idx = [int(Decimal(str(ix)).quantize(Decimal("0"), rounding=ROUND_HALF_UP)) for ix in idx]
        features = np.array([])
        for col in signal.columns:
            ecdf = ECDF(signal[col].values)  # fit
            x = ecdf.x[1:]  # Remove -inf
            feat = x[idx]
            features = np.hstack([features, feat])

        return features

    def obtain_mean(self, signal) -> np.ndarray:
        return signal.mean().values

    def obtain_std(self, signal) -> np.ndarray:
        return signal.std().values

    def obtain_mad(self, signal) -> np.ndarray:
        return stats.median_absolute_deviation(signal, axis=0)

    def obtain_max(self, signal) -> np.ndarray:
        return signal.max().values

    def obtain_min(self, signal) -> np.ndarray:
        return signal.min().values

    def obtain_sma(self, signal, window_size=128) -> np.ndarray:
        window_second = window_size / self.fs
        return sum(signal.sum().values - self.obtain_min(signal) * len(signal)) / window_second

    def obtain_energy(self, signal) -> np.ndarray:
        return norm(signal, ord=2, axis=0) ** 2 / len(signal)

    def obtain_iqr(self, signal) -> np.ndarray:
        return signal.quantile(0.75).values - signal.quantile(0.25).values

    def obtain_entropy(self, signal) -> np.ndarray:
        signal = signal - signal.min()
        return stats.entropy(signal)

    def obtain_arCoeff(self, signal) -> np.ndarray:
        arCoeff = np.array([])
        for col in signal.columns:
            val, _ = burg(signal[col], order=4)
            arCoeff = np.hstack((arCoeff, val))
        return arCoeff

    def obtain_correlation(self, signal) -> np.ndarray:
        if signal.shape[1] == 1:  # Signal dimension is 1
            correlation = np.array([])
        else:  # Signal dimension is 3
            xy = np.corrcoef(signal["x"], signal["y"])[0][1]
            yz = np.corrcoef(signal["y"], signal["z"])[0][1]
            zx = np.corrcoef(signal["z"], signal["x"])[0][1]
            correlation = np.hstack((xy, yz, zx))
        return correlation

    def obtain_maxInds(self, signal) -> np.ndarray:
        return signal.idxmax().values

    def obtain_meanFreq(self, signal) -> np.ndarray:
        meanFreq = np.array([])
        for col in signal.columns:
            val = np.mean(signal[col] * np.arange(len(signal)))
            meanFreq = np.hstack((meanFreq, val))
        return meanFreq

    def obtain_skewness(self, signal) -> np.ndarray:
        return signal.skew().values

    def obtain_kurtosis(self, signal) -> np.ndarray:
        return signal.kurt().values

    def obtain_bandsEnergy(self, signal) -> np.ndarray:
        bandsEnergy = np.array([])
        bins = [0, 4, 8, 12, 16, 20, 24, 29, 34, 39, 44, 49, 54, 59, 64]
        for i in range(len(bins) - 1):
            df = signal.iloc[bins[i] : bins[i + 1]]
            arr = self.obtain_energy(df)
            bandsEnergy = np.hstack((bandsEnergy, arr))
        return bandsEnergy

    def obtain_angle(self, v1, v2) -> np.ndarray:
        length = lambda v: math.sqrt(np.dot(v, v))
        return math.acos(np.dot(v1, v2) / (length(v1) * length(v2)))

def active_matrix_from_angle(basis, angle):
    """Compute active rotation matrix from rotation about basis vector.
    Parameters
    ----------
    basis : int from [0, 1, 2]
        The rotation axis (0: x, 1: y, 2: z)
    angle : float
        Rotation angle
    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    c = np.cos(angle)
    s = np.sin(angle)
    rep_time = angle.shape[0]

    if basis == 0:
        # R = np.array([[1.0, 0.0, 0.0],
        #               [0.0, c, -s],
        #               [0.0, s, c]])
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0., 0.],
                      [0.0, 0., 0.]])
        R = np.expand_dims(R,0).repeat(rep_time,axis=0)
        R[:,1,1] = c
        R[:,1,2] = -s
        R[:,2,1] = s
        R[:,2,2] = c
    elif basis == 1:
        # R = np.array([[c, 0.0, s],
        #               [0.0, 1.0, 0.0],
        #               [-s, 0.0, c]])
        R = np.array([[0., 0.0, 0.],
                      [0.0, 1.0, 0.0],
                      [0., 0.0, 0.]])
        R = np.expand_dims(R,0).repeat(rep_time,axis=0)
        R[:,0,0] = c
        R[:,0,2] = s
        R[:,2,0] = -s
        R[:,2,2] = c
    elif basis == 2:
        # R = np.array([[c, -s, 0.0],
        #               [s, c, 0.0],
        #               [0.0, 0.0, 1.0]])
        R = np.array([[0., 0., 0.0],
                      [0., 0., 0.0],
                      [0.0, 0.0, 1.0]])
        R = np.expand_dims(R,0).repeat(rep_time,axis=0)
        R[:,0,0] = c
        R[:,0,1] = -s
        R[:,1,0] = s
        R[:,1,1] = c
    else:
        raise ValueError("Basis must be in [0, 1, 2]")

    return R

def active_matrix_from_extrinsic_euler_xyz(e):
    """Compute active rotation matrix from extrinsic xyz Cardan angles.
    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, y-, and z-axes (extrinsic rotations)
    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = np.matmul(active_matrix_from_angle(0, alpha), active_matrix_from_angle(1, beta))
    R = np.matmul(active_matrix_from_angle(2, gamma), R)
    # R = active_matrix_from_angle(2, gamma).dot(
    #     active_matrix_from_angle(1, beta)).dot(
    #     active_matrix_from_angle(0, alpha))
    return R

def correct_orientation9(acc_grav, acc_body, gyro_raw, orientation):
    
    # cal the orientation correction mat numpy
    orientation_x      = orientation[:,1]
    orientation_y      = orientation[:,2]
    orientation_z      = orientation[:,0]
    orientation_xyz    = np.array([orientation_x, orientation_y, orientation_z])
    correct_R          = active_matrix_from_extrinsic_euler_xyz(orientation_xyz)
    # correct the orientation of grav, acc and gyro
    grav_xyz           = acc_grav.values
    lacc_xyz           = acc_body.values
    gyro_xyz           = gyro_raw.values
    grav_xyz           = np.matmul(correct_R, np.expand_dims(grav_xyz,2)).squeeze()
    lacc_xyz           = np.matmul(correct_R, np.expand_dims(lacc_xyz,2)).squeeze()
    gyro_xyz           = np.matmul(correct_R, np.expand_dims(gyro_xyz,2)).squeeze()
    # transform to DataFrame format
    grav_xyz           = pd.DataFrame(grav_xyz, columns = ["x", "y", "z"])
    lacc_xyz           = pd.DataFrame(lacc_xyz, columns = ["x", "y", "z"])
    gyro_xyz           = pd.DataFrame(gyro_xyz, columns = ["x", "y", "z"])
    return grav_xyz, lacc_xyz, gyro_xyz

def correct_orientation6(acc_raw, gyro_raw, orientation):
    
    # cal the orientation correction mat numpy
    orientation_x      = orientation[:,1]
    orientation_y      = orientation[:,2]
    orientation_z      = orientation[:,0]
    orientation_xyz    = np.array([orientation_x, orientation_y, orientation_z])
    correct_R          = active_matrix_from_extrinsic_euler_xyz(orientation_xyz)
    # correct the orientation of grav, acc and gyro
    acc_xyz            = acc_raw.values
    gyro_xyz           = gyro_raw.values
    acc_xyz            = np.matmul(correct_R, np.expand_dims(acc_xyz,2)).squeeze()
    gyro_xyz           = np.matmul(correct_R, np.expand_dims(gyro_xyz,2)).squeeze()
    # transform to DataFrame format
    acc_xyz            = pd.DataFrame(acc_xyz, columns = ["x", "y", "z"])
    gyro_xyz           = pd.DataFrame(gyro_xyz, columns = ["x", "y", "z"])
    return acc_xyz, gyro_xyz

def pre_threeaxis_data(to_NED, concat_data, R, scaler = "normalize"):
# X_data = pre_row_data(concat_data[i], std[i], cal_attitude_angle ,to_NED, ori_data_concat,ned_std,scaler = "normalize")
    
    if to_NED == True:
        data_all_axis_array = to_ned(concat_data,R)  #(40000, 3, 500) R:(40000,500,3,3)

    # data_all_axis_array = np.transpose(concat_data, (0, 2, 1))  #(40000, 500, 3)
    # data_shape = data_all_axis_array.shape

    # data_all_axis_array = data_all_axis_array.reshape(-1, data_shape[-1])

    # data_all_axis_array = pd.DataFrame(
    #     data_all_axis_array,
    #     columns=['x', 'y', 'z']
    # )

    # data_all_axis_array = scale(data_all_axis_array, scaler)
    # data_all_axis_array = data_all_axis_array.values  # 返回给定字典中可用的所有值的列表
    # data_all_axis_array = data_all_axis_array.reshape(data_shape[0],
    #                                                   data_shape[1],
    #                                                   data_shape[2])
    # data_all_axis_array = np.transpose(data_all_axis_array, (0, 2, 1))
    # (40000, 3, 500)
    
    return data_all_axis_array

def NED_R(ori_data):

    ori_data_concat = np.expand_dims(ori_data,axis=2) # (500,4,1)

    qw = ori_data_concat[:, 0]
    qx = ori_data_concat[:, 1]
    qy = ori_data_concat[:, 2]
    qz = ori_data_concat[:, 3]

    one = np.ones((ori_data_concat.shape[0], 1))  # (500,1)

    R = one - (qy * qy + qz * qz) - (qy * qy + qz * qz)  # (500,1)  数组* 相当于相应元素点乘
    R = np.append(R, (qx * qy - qw * qz) + (qx * qy - qw * qz), axis=1)  # (500,2)
    R = np.append(R, (qx * qz + qw * qy) + (qx * qz + qw * qy), axis=1)
    R = np.append(R, (qx * qy + qw * qz) + (qx * qy + qw * qz), axis=1)
    R = np.append(R, one - (qx * qx + qz * qz) - (qx * qx + qz * qz), axis=1)
    R = np.append(R, (qy * qz - qw * qx) + (qy * qz - qw * qx), axis=1)
    R = np.append(R, (qx * qz - qw * qy) + (qx * qz - qw * qy), axis=1)
    R = np.append(R, (qy * qz + qw * qx) + (qy * qz + qw * qx), axis=1)
    R = np.append(R, one - (qy * qy + qx * qx) - (qy * qy + qx * qx), axis=1)
    # R (40000,500,9)
    R  = R.reshape([-1,3,3])
    # R (500,3,3)

    return R


def to_ned(concat_data, R):
    # 读入已经转换成npy文件的转换的训练文件，读入给出的ori_w,x,y,z(quaternions),（x,y,z）= R（x,y,z）
    # 读入的npy文件是（16308，3，6000）
    # R = [16310,6000,3,3] 读入的传感器数据应该是[16300,6000,3,1],M = R * X
    # 由于内存原因，挑选一部分数据出来训练，目前选择[n=1800,6000,3]

    # R (40000,500,3,3)
    # concat_data(40000,3, 500)
    # concat_data = np.transpose(concat_data,[1,0])
    concat_data = np.expand_dims(concat_data, axis=2) # (500,3,1)
    # concat_ned_data = np.empty_like(concat_data) # (500,3,1)

    # for z in range(concat_data.shape[0]):
    #     for v in range(concat_data.shape[1]):
    #         concat_ned_data[z, v, :, :] = np.dot(R[z, v, :, :], concat_data[z, v, :, :])   #  矩阵乘法
    # concat_ned_data = concat_ned_data.squeeze()
    
    concat_ned_data = np.matmul(R, concat_data).squeeze()

    # return np.transpose(concat_ned_data,[0,2,1])
    return concat_ned_data