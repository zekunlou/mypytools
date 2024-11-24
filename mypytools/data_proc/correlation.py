import numpy




def auto_correlation(
    data:numpy.ndarray=None,
    max_lag:int=0,
    allow_memory:bool=True,
):
    """
    https://en.wikipedia.org/wiki/Autocorrelation#Estimation

    Params:
        - data: numpy.ndarray, shape (n_traj, time_series) or (time_series,)
        - max_lag: int, maximum lag to compute the auto-correlation

    Return:
        auto_correlation, shape (n_traj, max_lag)

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



