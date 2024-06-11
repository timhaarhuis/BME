import csv
import time
from CreaTeBME import SensorManager
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

manager = SensorManager(['0BE6'])  # Change 0BE6 to your sensor name
manager.set_sample_rate(100)
manager.start()

# Data lists
acc_x_data, acc_y_data, acc_z_data = [], [], []
gyr_x_data, gyr_y_data, gyr_z_data = [], [], []
angles = []

# High-pass filter function for gyroscope data
def high_pass_filter(data, alpha):
    filtered_data = [0] * len(data)
    for i in range(1, len(data)):
        filtered_data[i] = alpha * (filtered_data[i-1] + data[i] - data[i-1])
    return filtered_data

# Low-pass filter function for accelerometer data
def low_pass_filter(data, alpha):
    filtered_data = [0] * len(data)
    filtered_data[0] = data[0]  # Initialize with the first value
    for i in range(1, len(data)):
        filtered_data[i] = alpha * filtered_data[i-1] + (1 - alpha) * data[i]
    return filtered_data

# Function to read and process sensor data with complementary filter
def read_sensor_data(delay=0.01, alpha=0.98):
    global acc_x_data, acc_y_data, acc_z_data
    global gyr_x_data, gyr_y_data, gyr_z_data
    global angles, dt

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

    if not acc_y_data or not acc_z_data:
        print("No data available from the sensor.")
        return

    # Initial angle (assuming starting at rest)
    theta = 0.0
    dt = 0.01  # Assuming a sampling rate of 100 Hz

    # Apply low-pass filter to accelerometer data to calculate theta_accel
    acc_angle_data = [np.arctan2(y, z) * (180 / np.pi) for y, z in zip(acc_y_data, acc_z_data)]
    theta_accel_filtered = low_pass_filter(acc_angle_data, alpha)

    # Apply high-pass filter to gyroscope data to calculate theta_gyro
    gyro_rate_data = [x * dt for x in gyr_x_data]
    theta_gyro_filtered = high_pass_filter(gyro_rate_data, alpha)
    theta_gyro_filtered = np.cumsum(theta_gyro_filtered)  # Integrate the high-pass filtered rates to get angles

    # Combine filtered gyroscope and accelerometer data using the complementary filter
    for theta_gyro, theta_accel in zip(theta_gyro_filtered, theta_accel_filtered):
        theta = alpha * theta_gyro + (1 - alpha) * theta_accel
        theta = theta * 2
        angles.append(theta)

# Setup the plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

def update_plot(frame):
    read_sensor_data()

    # Limit data lists to the last 1000 samples for better visualization
    max_length = 1000
    acc_x_data_trimmed = acc_x_data[-max_length:]
    acc_y_data_trimmed = acc_y_data[-max_length:]
    acc_z_data_trimmed = acc_z_data[-max_length:]
    gyr_x_data_trimmed = gyr_x_data[-max_length:]
    gyr_y_data_trimmed = gyr_y_data[-max_length:]
    gyr_z_data_trimmed = gyr_z_data[-max_length:]
    angles_trimmed = angles[-max_length:]

    ax1.clear()
    ax2.clear()
    ax3.clear()

    # Accelerometer data plot
    ax1.plot(acc_x_data_trimmed, label='acc_x')
    ax1.plot(acc_y_data_trimmed, label='acc_y')
    ax1.plot(acc_z_data_trimmed, label='acc_z')
    ax1.set_title('Accelerometer Data')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Acceleration')
    ax1.legend()

    # Gyroscope data plot
    ax2.plot(gyr_x_data_trimmed, label='gyr_x')
    ax2.plot(gyr_y_data_trimmed, label='gyr_y')
    ax2.plot(gyr_z_data_trimmed, label='gyr_z')
    ax2.set_title('Gyroscope Data')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Angular Rate')
    ax2.legend()

    # Filtered angle plot
    ax3.plot(angles_trimmed, label='Filtered Angle')
    ax3.set_title('Filtered Angle using Complementary Filter')
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Angle (degrees)')
    ax3.legend()

ani = animation.FuncAnimation(fig, update_plot, interval=100)
plt.tight_layout()
plt.show()
