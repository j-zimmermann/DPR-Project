import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, convolve


def DPR_UpdateSingle(I_in, PSF, gain, window_radius):
    # Set parameter for image upscaling
    # Convert PSF to the 1/e radius
    PSF = PSF / 1.6651

    # Upscaled input image
    number_row_initial, number_column_initial = I_in.shape
    x0 = np.linspace(-0.5, 0.5, number_column_initial)
    y0 = np.linspace(-0.5, 0.5, number_row_initial)

    x = np.linspace(-0.5, 0.5, round(5 * number_column_initial / PSF))
    y = np.linspace(-0.5, 0.5, round(5 * number_row_initial / PSF))
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Set the Sobel kernel
    sobelX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # DPR on single frames
    single_frame_I_in = I_in - np.min(I_in)

    local_minimum = np.zeros((number_row_initial, number_column_initial))
    single_frame_I_in_localmin = np.zeros((number_row_initial, number_column_initial))

    for u in range(number_row_initial):
        for v in range(number_column_initial):
            # define sub window - window size: x times 1/e radius
            sub_window = single_frame_I_in[max(1, u - window_radius):min(number_row_initial, u + window_radius),
                                           max(1, v - window_radius):min(number_column_initial, v + window_radius)]
            # find local minimum in the local window
            local_minimum[u, v] = np.min(sub_window)
            # I - local_min(I)
            single_frame_I_in_localmin[u, v] = single_frame_I_in[u, v] - local_minimum[u, v]

    # upscale
    # I - local min - used to calculate gradient
    interpolator = RegularGridInterpolator((x0, y0), single_frame_I_in_localmin, method='cubic', fill_value=0)
    single_frame_localmin_magnified = interpolator((X, Y))
    single_frame_localmin_magnified[single_frame_localmin_magnified < 0] = 0
    single_frame_localmin_magnified = np.pad(single_frame_localmin_magnified, ((10, 10), (10, 10)), mode='constant', constant_values=0)

    # Raw image
    interpolator = RegularGridInterpolator((x0, y0), single_frame_I_in, method='cubic', fill_value=0)
    single_frame_I_magnified = interpolator((X, Y))
    single_frame_I_magnified[single_frame_I_magnified < 0] = 0
    single_frame_I_magnified = np.pad(single_frame_I_magnified, ((10, 10), (10, 10)), mode='constant', constant_values=0)

    number_row, number_column = single_frame_I_magnified.shape

    # locally normalized version of Im
    I_normalized = single_frame_localmin_magnified / (gaussian_filter(single_frame_localmin_magnified, sigma=10) + 0.00001)

    # calculate normalized gradients
    gradient_y = convolve(I_normalized, sobelX, mode='nearest')
    gradient_x = convolve(I_normalized, sobelY, mode='nearest')

    # calculate pixel displacements
    gain_value = 0.5 * gain + 1
    displacement_x = gain_value * gradient_x / (I_normalized + 0.00001)
    displacement_y = gain_value * gradient_y / (I_normalized + 0.00001)
    displacement_x[np.abs(displacement_x) > 10] = 0  # limit displacements to twice PSF size
    displacement_y[np.abs(displacement_y) > 10] = 0

    # calculate I_out with weighted pixel displacements
    single_frame_I_out = np.zeros((number_row, number_column))
    for nx in range(11, number_row - 10):
        for ny in range(11, number_column - 10):
            weighted1 = (1 - abs(displacement_x[nx, ny] - int(displacement_x[nx, ny]))) * (1 - abs(displacement_y[nx, ny] - int(displacement_y[nx, ny])))
            weighted2 = (1 - abs(displacement_x[nx, ny] - int(displacement_x[nx, ny]))) * (abs(displacement_y[nx, ny] - int(displacement_y[nx, ny])))
            weighted3 = (abs(displacement_x[nx, ny] - int(displacement_x[nx, ny]))) * (1 - abs(displacement_y[nx, ny] - int(displacement_y[nx, ny])))
            weighted4 = (abs(displacement_x[nx, ny] - int(displacement_x[nx, ny]))) * (abs(displacement_y[nx, ny] - int(displacement_y[nx, ny])))
            coordinate1 = [int(displacement_x[nx, ny]), int(displacement_y[nx, ny])]
            coordinate2 = [int(displacement_x[nx, ny]), int(displacement_y[nx, ny] + (displacement_y[nx, ny] > 0))]
            coordinate3 = [int(displacement_x[nx, ny] + (displacement_x[nx, ny] > 0)), int(displacement_y[nx, ny])]
            coordinate4 = [int(displacement_x[nx, ny] + (displacement_x[nx, ny] > 0)), int(displacement_y[nx, ny] + (displacement_y[nx, ny] > 0))]

            # Shift I-local_min, use 'single_frame_localmin_magnified',
            # shift raw image, use 'single_frame_I_magnified'
            single_frame_I_out[nx + coordinate1[0], ny + coordinate1[1]] = single_frame_I_out[nx + coordinate1[0], ny + coordinate1[1]] + weighted1 * single_frame_I_magnified[nx, ny]
            single_frame_I_out[nx + coordinate2[0], ny + coordinate2[1]] = single_frame_I_out[nx + coordinate2[0], ny + coordinate2[1]] + weighted2 * single_frame_I_magnified[nx, ny]
            single_frame_I_out[nx + coordinate3[0], ny + coordinate3[1]] = single_frame_I_out[nx + coordinate3[0], ny + coordinate3[1]] + weighted3 * single_frame_I_magnified[nx, ny]
            single_frame_I_out[nx + coordinate4[0], ny + coordinate4[1]] = single_frame_I_out[nx + coordinate4[0], ny + coordinate4[1]] + weighted4 * single_frame_I_magnified[nx, ny]

    single_frame_I_out = single_frame_I_out[11:-10, 11:-10]
    single_frame_I_magnified = single_frame_I_magnified[11:-10, 11:-10]

    return single_frame_I_out, single_frame_localmin_magnified
