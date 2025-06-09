import warnings
from typing import Optional, Union

import numpy
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def multiple_gaussian_functions(
    x: numpy.ndarray,
    mu: numpy.ndarray,
    sigma: numpy.ndarray,
    amp: numpy.ndarray,
):
    """Calculate multiple Gaussian function values for given parameters.

    This function computes the Gaussian (normal) distribution function:
    f(x) = amp * exp(-(x-mu)^2/(2*sigma^2))

    Args:
        x (numpy.ndarray): Input array for which to calculate Gaussian values.
        mu (numpy.ndarray): Means of the Gaussian distributions.
            Shape should be compatible with x, e.g. (n,) or (n, m).
        sigma (numpy.ndarray): Standard deviations of the Gaussian distributions.
            Should match mu in shape.
        amp (numpy.ndarray): Amplitudes of the Gaussian distributions.
            Should match mu in shape.

    Notice:
        mu and sigma and amp should be 1-dimensional arrays, and have the same shape.

    Returns:
        numpy.ndarray: Sum of Gaussian function values.
            Shape matches x.shape
    """
    assert mu.ndim == 1 and sigma.ndim == 1 and amp.ndim == 1, "mu, sigma, amp should be 1D arrays"
    assert mu.shape == sigma.shape == amp.shape, "mu, sigma, amp should have the same shape"
    assert x.ndim == 1, "x should be a 1D array"

    # Vectorized computation using broadcasting
    # x.shape: (m,), mu.shape: (n,) -> (m, n) after broadcasting
    x_expanded = x[:, numpy.newaxis]  # Shape: (m, 1)
    mu_expanded = mu[numpy.newaxis, :]  # Shape: (1, n)
    sigma_expanded = sigma[numpy.newaxis, :]  # Shape: (1, n)
    amp_expanded = amp[numpy.newaxis, :]  # Shape: (1, n)

    # Compute Gaussian for each component
    gaussian_components = amp_expanded * numpy.exp(-0.5 * ((x_expanded - mu_expanded) / sigma_expanded) ** 2)

    # Sum over all Gaussian components
    return numpy.sum(gaussian_components, axis=1)


def _objective_function(x: numpy.ndarray, *params) -> numpy.ndarray:
    """
    Objective function for curve fitting.

    Args:
        x: Input x values
        *params: Flattened parameters [amp1, mu1, sigma1, amp2, mu2, sigma2, ...]

    Returns:
        Sum of Gaussian functions
    """
    n_gaussians = len(params) // 3
    params_array = numpy.array(params).reshape(n_gaussians, 3)

    amp = params_array[:, 0]
    mu = params_array[:, 1]
    sigma = numpy.abs(params_array[:, 2])  # Ensure positive sigma

    return multiple_gaussian_functions(x, mu, sigma, amp)


def _estimate_initial_parameters(
    data_x: numpy.ndarray,
    data_y: numpy.ndarray,
    n_gaussians: int,
    mus: Optional[Union[float, list[float]]] = None,
    sigmas: Optional[Union[float, list[float]]] = None,
) -> numpy.ndarray:
    """
    Estimate initial parameters for Gaussian fitting.

    Returns:
        numpy.ndarray: Initial parameters flattened as [amp1, mu1, sigma1, amp2, mu2, sigma2, ...]
    """
    # Estimate initial mu values
    if mus is None:
        # Try to find peaks in the data
        try:
            peaks, properties = find_peaks(
                data_y, height=numpy.max(data_y) * 0.1, distance=len(data_x) // (n_gaussians * 2)
            )
            if len(peaks) >= n_gaussians:
                initial_mus = data_x[peaks[:n_gaussians]]
            else:
                # Fall back to evenly spaced values
                initial_mus = numpy.linspace(data_x.min(), data_x.max(), n_gaussians)
        except Exception:
            initial_mus = numpy.linspace(data_x.min(), data_x.max(), n_gaussians)
    elif isinstance(mus, (int, float)):
        initial_mus = numpy.full(n_gaussians, mus)
    else:
        initial_mus = numpy.array(mus[:n_gaussians])
        if len(initial_mus) < n_gaussians:
            # Pad with evenly spaced values
            remaining = numpy.linspace(data_x.min(), data_x.max(), n_gaussians - len(initial_mus))
            initial_mus = numpy.concatenate([initial_mus, remaining])

    # Estimate initial sigma values
    if sigmas is None:
        # Use a fraction of the data range as initial guess
        data_range = data_x.max() - data_x.min()
        initial_sigmas = numpy.full(n_gaussians, data_range / (n_gaussians * 4))
    elif isinstance(sigmas, (int, float)):
        initial_sigmas = numpy.full(n_gaussians, sigmas)
    else:
        initial_sigmas = numpy.array(sigmas[:n_gaussians])
        if len(initial_sigmas) < n_gaussians:
            data_range = data_x.max() - data_x.min()
            default_sigma = data_range / (n_gaussians * 4)
            remaining = numpy.full(n_gaussians - len(initial_sigmas), default_sigma)
            initial_sigmas = numpy.concatenate([initial_sigmas, remaining])

    # Estimate initial amplitudes
    max_y = numpy.max(data_y)
    initial_amps = numpy.full(n_gaussians, max_y / n_gaussians)

    # For peaks found, use actual peak heights
    if mus is None:
        try:
            peaks, _ = find_peaks(data_y, height=numpy.max(data_y) * 0.1)
            if len(peaks) > 0:
                for i, mu in enumerate(initial_mus[: len(peaks)]):
                    # Find closest peak
                    closest_peak_idx = numpy.argmin(numpy.abs(data_x[peaks] - mu))
                    if closest_peak_idx < len(peaks):
                        initial_amps[i] = data_y[peaks[closest_peak_idx]]
        except Exception:
            pass

    # Flatten parameters for curve_fit
    initial_params = numpy.column_stack([initial_amps, initial_mus, initial_sigmas]).flatten()

    return initial_params


def fit_with_gaussian(
    data_x: numpy.ndarray,
    data_y: numpy.ndarray,
    n_gaussians: int,
    mus: Optional[Union[float, list[float]]] = None,
    sigmas: Optional[Union[float, list[float]]] = None,
) -> tuple[numpy.ndarray, numpy.ndarray, float]:
    """
    Fit a spectrum with multiple Gaussian functions.

    Args:
        data_x: X-coordinates of the data points
        data_y: Y-coordinates of the data points
        n_gaussians: Number of Gaussian functions to fit.
        mus: Initial guess for the means of the Gaussian functions. If None, will be set to evenly spaced values in the
            range of data_x, or estimated from peaks if possible.
        sigmas: Initial guess for the standard deviations of the Gaussian functions. If None, will be set to a small
            value based on data range.

    Returns:
        fitted_params: numpy.ndarray: Fitted parameters for the Gaussian functions.
            Each row corresponds to a Gaussian function with [amplitude, mean, sigma].
        param_errors: numpy.ndarray: Estimated error for the fitted parameters.
            Same shape as fitted_params.
        rmse: float: Root Mean Square Error of the fit, for reference
    """
    assert data_x.ndim == 1 and data_y.ndim == 1, "data_x and data_y should be 1D arrays"
    assert len(data_x) == len(data_y), "data_x and data_y should have the same length"
    assert n_gaussians > 0, "n_gaussians should be positive"

    # Remove any NaN or infinite values
    valid_mask = numpy.isfinite(data_x) & numpy.isfinite(data_y)
    data_x_clean = data_x[valid_mask]
    data_y_clean = data_y[valid_mask]

    if len(data_x_clean) < 3 * n_gaussians:
        raise ValueError("Not enough data points for fitting. Need at least 3 points per Gaussian.")

    # Get initial parameter estimates
    initial_params = _estimate_initial_parameters(data_x_clean, data_y_clean, n_gaussians, mus, sigmas)

    # Set bounds to ensure physical constraints
    # Lower bounds: [amp_min, mu_min, sigma_min, ...]
    # Upper bounds: [amp_max, mu_max, sigma_max, ...]
    lower_bounds = []
    upper_bounds = []

    x_min, x_max = data_x_clean.min(), data_x_clean.max()
    _, y_max = data_y_clean.min(), data_y_clean.max()
    x_range = x_max - x_min

    for _ in range(n_gaussians):
        # Amplitude bounds
        lower_bounds.extend([0, x_min - x_range, 1e-6])  # Positive amplitude, flexible mu, small positive sigma
        upper_bounds.extend([y_max * 2, x_max + x_range, x_range])  # Reasonable upper limits

    try:
        # Perform the fit with error estimation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            popt, pcov = curve_fit(
                _objective_function,
                data_x_clean,
                data_y_clean,
                p0=initial_params,
                bounds=(lower_bounds, upper_bounds),
                maxfev=10000,
                method="trf",  # Trust Region Reflective algorithm, good for bounded problems
            )

        # Reshape fitted parameters
        fitted_params = popt.reshape(n_gaussians, 3)
        fitted_params[:, 2] = numpy.abs(fitted_params[:, 2])  # Ensure positive sigma

        # Calculate parameter errors
        if pcov is not None and numpy.all(numpy.isfinite(pcov)):
            param_std = numpy.sqrt(numpy.diag(pcov))
            param_errors = param_std.reshape(n_gaussians, 3)
        else:
            param_errors = numpy.full_like(fitted_params, numpy.nan)
            warnings.warn("Could not estimate parameter uncertainties.", stacklevel=2)

        # Calculate RMSE
        y_fitted = _objective_function(data_x_clean, *popt)
        rmse = numpy.sqrt(numpy.mean((data_y_clean - y_fitted) ** 2))

        # Sort by peak position (mu) for consistent output
        sort_indices = numpy.argsort(fitted_params[:, 1])
        fitted_params = fitted_params[sort_indices]
        param_errors = param_errors[sort_indices]

        return fitted_params, param_errors, rmse

    except Exception as e:
        raise RuntimeError(f"Fitting failed: {str(e)}. Try adjusting initial parameters or reducing n_gaussians.") \
            from e


# Example usage and testing function
def test_gaussian_fitting():
    """Test the Gaussian fitting functions with synthetic data."""
    # Generate synthetic data
    x = numpy.linspace(0, 10, 1000)

    # True parameters: [amplitude, mean, sigma] for each Gaussian
    true_params = numpy.array([[1.0, 2.5, 0.5], [1.5, 5.0, 0.8], [0.8, 7.5, 0.3]])

    # Generate synthetic spectrum
    y_true = multiple_gaussian_functions(x, true_params[:, 1], true_params[:, 2], true_params[:, 0])

    # Add noise
    numpy.random.seed(42)
    noise = numpy.random.normal(0, 0.05, len(y_true))
    y_noisy = y_true + noise

    # Fit the data
    fitted_params, param_errors, rmse = fit_with_gaussian(x, y_noisy, n_gaussians=3)

    print("True parameters:")
    print(true_params)
    print("\nFitted parameters:")
    print(fitted_params)
    print(f"\nRMSE: {rmse:.6f}")

    # Test the fitted model
    y_fitted = multiple_gaussian_functions(x, fitted_params[:, 1], fitted_params[:, 2], fitted_params[:, 0])

    return x, y_noisy, y_fitted, fitted_params, param_errors, rmse


if __name__ == "__main__":
    test_gaussian_fitting()
