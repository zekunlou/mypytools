from typing import Optional

import numpy


def time_correlation_function_by_fft_v1(
    data1: numpy.ndarray,
    data2: Optional[numpy.ndarray] = None,
    max_lag: Optional[int] = -1,
):
    """Calculate the time correlation function using the FFT method.

    Args:
        data1 (numpy.ndarray): First time series data, should be 1D or 2D. The last axis is the time axis.
        data2 (Optional[numpy.ndarray], optional): Second time series data. Defaults to None.
        max_lag (Optional[int], optional): Maximum correlation time lag in the output. Defaults to -1, which means full output.

    Returns:
        numpy.ndarray: Time correlation function, 1D or 2D based on the input data dimensions.
    """
    if data1.ndim == 1:
        data1 = data1.reshape(1, -1)
        if_input_data1_1d = True
    elif data1.ndim == 2:
        pass
        if_input_data1_1d = False
    else:
        raise ValueError(f"data1 must be 1D or 2D array, but got {data1.ndim=} and {data1.shape=}")
    data1 -= data1.mean(axis=-1, keepdims=True)
    data1_ft = numpy.fft.rfft(data1, axis=-1, n=2*data1.shape[-1])  # pad 0 for full FFT
    # data1_ft = numpy.fft.rfft(data1, axis=-1)
    # data1_ft[:, 0] = 0  # remove the mean value of the data

    if data2 is None:
        data2_ft = data1_ft
    else:
        if data2.ndim == 1:
            data2 = data2.reshape(1, -1)
        elif data2.ndim == 2:
            assert data1.shape[0] == data2.shape[0], \
                f"{data1.shape=} and {data2.shape=}, the first dimension must be the same"
        else:
            raise ValueError(f"data2 must be 1D or 2D array, but got {data2.ndim=} and {data2.shape=}")
        data2 -= data2.mean(axis=-1, keepdims=True)
        data2_ft = numpy.fft.rfft(data2, axis=-1, n=2*data2.shape[-1])  # pad 0 for full FFT
        # data2_ft = numpy.fft.rfft(data2, axis=-1)
        # data2_ft[:, 0] = 0  # remove the mean value of the data

    max_ft_length = max(data1_ft.shape[-1], data2_ft.shape[-1])
    autocorr_ft = data1_ft.conj()[:,:max_ft_length] * data2_ft[:,:max_ft_length]
    if data2 is None:
        autocorr = numpy.fft.irfft(autocorr_ft.real, axis=-1)
    else:
        autocorr = numpy.fft.ifft(autocorr_ft, axis=-1)

    if data2 is None:
        normalizer = data1.std(axis=-1)**2 * data1.shape[-1]
    else:
        normalizer = data1.std(axis=-1) * data2.std(axis=-1) * (data1.shape[-1] * data2.shape[-1])**0.5
    autocorr = autocorr / normalizer.reshape(-1,1)

    if max_lag != -1:
        assert max_lag > 0
        max_lag = min(max_lag+1, autocorr.shape[-1])
        autocorr = autocorr[..., :max_lag]
    if if_input_data1_1d:
        autocorr = autocorr[0]

    return autocorr



def autocorrelate_from_howtos(A: numpy.ndarray, axis: int = 0) -> numpy.ndarray:
    """
    Compute the autocorrelation function of an array along the specified axis.

    :param A: array containing the time series
    :type A: numpy.ndarray
    :param axis: axis corresponding to time, defaults to 0
    :type axis: Optional[int], optional
    :return: autocorrelation function of the array input
    :rtype: numpy.ndarray
    """
    axis = axis % A.ndim
    len_a = A.shape[axis]
    len_fft = 2 * len_a

    # Normalization factor (decreasing window size)
    norm_tcf = numpy.arange(len_a, 0, -1, dtype=int)

    # Expand normalization dimensions
    norm_tcf = numpy.expand_dims(norm_tcf, tuple(i for i in range(A.ndim) if i != axis))

    # Compute FFT of A
    ftA = numpy.fft.rfft(A, axis=axis, n=len_fft)

    # Power spectrum (multiply by complex conjugate)
    ftA *= numpy.conj(ftA)

    # Inverse FFT to get the auto-correlation
    autocorr = numpy.fft.irfft(ftA, axis=axis, n=len_fft)

    # Slice the auto-correlation to original signal length
    autocorr = numpy.take(autocorr, numpy.arange(len_a), axis=axis)

    return autocorr / norm_tcf



def cal_auto_correlation(
    data:numpy.ndarray,
    max_lag:int=0,
    allow_memory:bool=True,
):
    """
    https://en.wikipedia.org/wiki/Autocorrelation#Estimation

    Args:
        data: numpy.ndarray, shape (n_traj, time_series) or (time_series,)
        max_lag: int, maximum lag to compute the auto-correlation
        allow_memory: bool, whether to use memory-heavy version

    Return:
        numpy.ndarray: auto_correlation, shape (n_traj, max_lag) or (max_lag,) depending on the input data shape

    Definition:
    $$ C(j) = \frac{1}{(N-j) \sigma^2} \sum_{i=0}^{N-j-1} (x_i - \mu)(x_{i+j} - \mu) $$
    and $ \mu = \sum_{i=0}^{N-1} x_i / N $, $ \sigma^2 = \sum_{i=0}^{N-1} (x_i - \mu)^2 / N $.
    As definition, $ C(0) = 1 $
    """

    if data.ndim == 1:
        data = data.reshape(1, -1)
        data_dim = 1
    else:
        assert data.ndim == 2, f"Expect data.ndim == 2, got {data.ndim}"
        data_dim = 2
    """now data.shape = (n_traj, time_series)"""
    mu = numpy.mean(data, axis=-1, keepdims=True)
    sigma2 = numpy.var(data, axis=-1, keepdims=True)
    data_unbiased = data - mu

    if allow_memory:
        """the memory-heavy version"""
        data_rolled = []
        for lag in range(max_lag+1):
            _tmp = numpy.roll(data_unbiased, lag, axis=-1)
            _tmp[..., :lag] = 0
            data_rolled.append(_tmp)
        data_rolled = numpy.stack(data_rolled, axis=-1)  # shape (n_traj, time_series, max_lag+1)
        auto_correlation = numpy.mean(data_unbiased[:,:,None] * data_rolled, axis=-2) / sigma2
    else:
        """process by each lag, but batched by n_traj"""
        auto_correlation = numpy.zeros((data.shape[0], max_lag+1))
        for lag in range(max_lag+1):
            auto_correlation[:, lag] = numpy.mean(
                data_unbiased[:, :-lag] * data_unbiased[:, lag:]
            ) / sigma2

    if data_dim == 1:
        assert auto_correlation.shape[0] == 1, f"Expect auto_correlation.shape[0] == 1, got {auto_correlation.shape[0]}"
        auto_correlation = auto_correlation.reshape(-1)
    return auto_correlation

def cal_auto_correlation_velocity(
    data:numpy.ndarray,
    max_lag:int=0,
):
    """calculate velocity auto-correlation.

    Args:
        data (numpy.ndarray): data array, shape (natoms, 3, time_series)
        max_lag (int): maximum lag to compute the auto-correlation

    Returns:
        numpy.ndarray: auto_correlation, shape (max_lag,)

    Definition:
    $$ C_{vv} (t) $$
    """

    assert data.ndim == 3, f"Expect data.ndim == 3, got {data.ndim}"
    mu = numpy.mean(data, axis=-1, keepdims=True)
    sigma2 = numpy.var(data, keepdims=True)
    data_unbiased = data - mu
    print(data_unbiased.shape, sigma2)

    """process by each lag"""
    auto_correlation = numpy.zeros(max_lag+1)
    for lag in range(max_lag+1):
        if lag == 0:
            auto_correlation[lag] = numpy.mean(data_unbiased * data_unbiased) / sigma2
        else:
            auto_correlation[lag] = numpy.mean(
                data_unbiased[:, :, :-lag] * data_unbiased[:, :, lag:]
            ) / sigma2

    return auto_correlation




def cal_auto_correlation_smooth(
    data:numpy.ndarray,
    seg_length:int,
    seg_shift:int,
    max_lag:int=0,
    allow_memory:bool=True,
    ac_func_str:callable="conv_simple",
    verbose:bool=False,
):
    """Calculate smoothed auto-correlation.
    Based on the function `auto_correlation`, one calculate the auto-cor in each segment with length `seg_length`
    and shift by `seg_shift`. Then the results are averaged.

    Args:
        data (numpy.ndarray): data array, shape (n_traj, time_series) or (time_series,)
        seg_length (int): the length of each segment for auto-cor calculation
        seg_shift (int): the shift between two segments
        max_lag (int, optional): maximum lag to compute the auto-correlation. Defaults to 0.
        allow_memory (bool, optional): whether to use memory-heavy version. Default to True.

    Returns:
        numpy.ndarray: auto_correlation, shape (n_traj, max_lag) or (max_lag,) depending on the input data shape
    """

    if data.ndim == 1:
        data = data.reshape(1, -1)
        data_dim = 1
    else:
        assert data.ndim == 2, f"Expect data.ndim == 2, got {data.ndim}"
        data_dim = 2

    ac_func_dict = {
        "conv_simple": cal_auto_correlation,
        "fft_v1": time_correlation_function_by_fft_v1,
        "how-tos_eg": autocorrelate_from_howtos,
    }
    assert ac_func_str in ac_func_dict.keys()

    """now data.shape = (n_traj, time_series)"""
    n_traj, time_series = data.shape
    n_seg = (time_series - seg_length) // seg_shift + 1
    seg_start_indices = numpy.arange(n_seg) * seg_shift
    if verbose:
        # print(f"{n_traj = }, {time_series = }")
        print(f"{n_seg = }, {seg_start_indices = }")
    assert seg_start_indices[-1] + seg_length <= time_series, \
        f"time series length mismatch, got {seg_start_indices[-1] + seg_length = } > {time_series = }"
    auto_correlation = numpy.zeros((n_traj, max_lag+1))
    for this_seg_start in seg_start_indices:
        data_seg = data[:, this_seg_start:(this_seg_start+seg_length)]
        if ac_func_str == "how-tos_eg":
            auto_correlation += (
                ac_func_dict[ac_func_str](
                    data_seg - data_seg.mean(axis=-1, keepdims=True),
                    axis=-1,
                ) / data_seg.std()**2
            )[..., :max_lag+1]
        elif ac_func_str == "conv_simple":
            auto_correlation += ac_func_dict[ac_func_str](data_seg, max_lag=max_lag, allow_memory=allow_memory)
        elif ac_func_str == "fft_v1":
            auto_correlation += ac_func_dict[ac_func_str](data_seg, max_lag=max_lag)
        else:
            raise ValueError(f"Unknown ac_func: {ac_func_str}")
    auto_correlation /= n_seg

    if data_dim == 1:
        assert auto_correlation.shape[0] == 1, f"Expect auto_correlation.shape[0] == 1, got {auto_correlation.shape[0]}"
        auto_correlation = auto_correlation.reshape(-1)
    return auto_correlation

