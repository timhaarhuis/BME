# Import the necessary packages
from CreaTeBME import SensorManager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import scipy.signal as signal
from scipy.signal import find_peaks

# Define sampling parameters
dt = 1/60
fs = 60
fc = 15
alpha = 0.98

def low_pass_filter(data, alpha=0.1):
    filtered_data = []
    for i, x in enumerate(data):
        if i == 0:
            filtered_data.append(x)
        else:
            filtered_data.append(alpha * x + (1 - alpha) * filtered_data[-1])
    return filtered_data

def high_pass_filter(data, alpha=0.1):
    filtered_data = []
    prev_filtered = data[0]
    for i, x in enumerate(data):
        if i == 0:
            filtered_data.append(x)
        else:
            filtered_value = alpha * (filtered_data[-1] + x - data[i-1])
            filtered_data.append(filtered_value)
            prev_filtered = filtered_value
    return filtered_data

# Sliding window average function
def sliding_window_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Outlier rejection filter
def outlier_rejection_filter(data, threshold):
    filtered_data = [data[0]]
    for i in range(1, len(data)):
        if abs(data[i] - data[i-1]) > threshold:
            filtered_data.append(filtered_data[-1])
        else:
            filtered_data.append(data[i])
    return filtered_data

# Set up the sensor manager
manager = SensorManager(['0BE6'])  # Change 0BE6 to your sensor name
manager.set_sample_rate(100)
manager.start()

# Data lists
acc_x_data, acc_y_data, acc_z_data = [], [], []
gyr_x_data, gyr_y_data, gyr_z_data = [], [], []
angles = []
dt = 1/100  # Assuming a sample rate of 100 Hz
outlier_threshold = 30  # Define a threshold for outlier rejection (angle change in degrees per sample)

# Initialize the plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

# Initialize plot lines
acc_x_line, = ax1.plot([], [], label='acc_x')
acc_y_line, = ax1.plot([], [], label='acc_y')
acc_z_line, = ax1.plot([], [], label='acc_z')

gyr_x_line, = ax2.plot([], [], label='gyr_x')
gyr_y_line, = ax2.plot([], [], label='gyr_y')
gyr_z_line, = ax2.plot([], [], label='gyr_z')

angle_line, = ax3.plot([], [], label='Filtered Angle')

# Setting up plot titles and labels
ax1.set_title('Accelerometer Data')
ax1.set_xlabel('Sample')
ax1.set_ylabel('Acceleration')
ax1.legend()

ax2.set_title('Gyroscope Data')
ax2.set_xlabel('Sample')
ax2.set_ylabel('Angular Rate')
ax2.legend()

ax3.set_title('Filtered Angle using Complementary Filter')
ax3.set_xlabel('Sample')
ax3.set_ylabel('Angle (degrees)')
ax3.legend()

def update(frame):
    global acc_x_data, acc_y_data, acc_z_data
    global gyr_x_data, gyr_y_data, gyr_z_data
    global angles, alpha, dt

    measurements = manager.get_measurements()
    for sensor, data in measurements.items():
        if len(data) > 0:
            for datum in data:
                acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z = datum

                acc_x_data.append(acc_x)
                acc_y_data.append(acc_y)
                acc_z_data.append(acc_z)
                gyr_x_data.append(gyr_x)
                gyr_y_data.append(gyr_y)
                gyr_z_data.append(gyr_z)

                if len(acc_x_data) > 1:
                    # Apply low-pass filter to accelerometer data
                    acc_angle = np.arctan2(acc_y, acc_z) * (180 / np.pi)
                    acc_angles_filtered = low_pass_filter([acc_angle], alpha=0.1)[0]

                    # Apply high-pass filter to gyroscope data
                    gyro_rate = gyr_x * dt
                    gyro_rates_filtered = high_pass_filter([gyro_rate], alpha=0.1)[0]
                    gyro_angles_filtered = np.cumsum([gyro_rates_filtered])[-1]  # Integrate the high-pass filtered rates

                    # Combine filtered gyroscope and accelerometer data
                    theta = alpha * gyro_angles_filtered + (1 - alpha) * acc_angles_filtered
                    theta = (theta + 360) % 360  # Normalize to range [0, 360)
                    if theta > 180:
                        theta -= 360  # Normalize to range [-180, 180)

                    angles.append(theta)

                # Apply outlier rejection filter
                if len(angles) > 1:
                    angles = outlier_rejection_filter(angles, outlier_threshold)

                # Apply sliding window average
                if len(angles) >= 10:  # Use a window size of 10 for example
                    angles_smoothed = sliding_window_average(angles, 10)
                else:
                    angles_smoothed = angles

                # Update plots
                acc_x_line.set_data(range(len(acc_x_data)), acc_x_data)
                acc_y_line.set_data(range(len(acc_y_data)), acc_y_data)
                acc_z_line.set_data(range(len(acc_z_data)), acc_z_data)

                gyr_x_line.set_data(range(len(gyr_x_data)), gyr_x_data)
                gyr_y_line.set_data(range(len(gyr_y_data)), gyr_y_data)
                gyr_z_line.set_data(range(len(gyr_z_data)), gyr_z_data)

                angle_line.set_data(range(len(angles_smoothed)), angles_smoothed)

                # Set plot limits if data is available
                if len(acc_x_data) > 1:
                    ax1.set_xlim(0, len(acc_x_data))
                    ax1.set_ylim(min(acc_x_data + acc_y_data + acc_z_data), max(acc_x_data + acc_y_data + acc_z_data))
                if len(gyr_x_data) > 1:
                    ax2.set_xlim(0, len(gyr_x_data))
                    ax2.set_ylim(min(gyr_x_data + gyr_y_data + gyr_z_data), max(gyr_x_data + gyr_y_data + gyr_z_data))
                if len(angles_smoothed) > 1:
                    ax3.set_xlim(0, len(angles_smoothed))
                    ax3.set_ylim(min(angles_smoothed), max(angles_smoothed))

    return acc_x_line, acc_y_line, acc_z_line, gyr_x_line, gyr_y_line, gyr_z_line, angle_line

# Use FuncAnimation to update the plot
ani = FuncAnimation(fig, update, interval=1000/100, cache_frame_data=False)

plt.tight_layout()
plt.show()

# Stop the sensor manager when done
manager.stop()
