from CreaTeBME import SensorEmulator, SensorManager
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import scipy.signal as signal

# Create a sensor manager for the given sensor names using the given callback
manager = SensorManager(['8732'])

sample_rate = 100
max_samples = 1000  # Maximum number of samples to store
accelerometer = np.zeros((max_samples, 3))  # Initialize numpy array for accelerometer data
gyroscope = np.zeros((max_samples, 3))  # Initialize numpy array for gyroscope data
angles = np.zeros(max_samples)  # Initialize numpy array for the complementary filter angle

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)  # Create subplots for accelerometer, gyroscope, and angle data

# Start the sensor manager
manager.start()
manager.set_sample_rate(sample_rate)

# Current index in the circular buffer
current_index = 0
dt = 1 / sample_rate  # Time step
alpha = 0.98  # Complementary filter coefficient
angle = 90  # Initial angle estimate (assuming the trunk starts upright at 90 degrees)

# Filter settings
fc = 15  # Cutoff frequency
nyq = 0.5 * sample_rate  # Nyquist frequency
normal_cutoff = fc / nyq  # Normalized cutoff frequency

# Define filter coefficients
b_low, a_low = signal.butter(2, normal_cutoff, btype='low', analog=False)
b_high, a_high = signal.butter(2, normal_cutoff, btype='high', analog=False)


def low_pass_filter(data):
    return signal.filtfilt(b_low, a_low, data)


def high_pass_filter(data):
    return signal.filtfilt(b_high, a_high, data)


def final_acc_values(data):

    filtered = low_pass_filter(data)



def animate(frame):
    global accelerometer, gyroscope, angles, current_index, angle

    measurements = manager.get_measurements()

    for sensor, data in measurements.items():
        if len(data) > 0:
            for datapoint in data:
                acc_x, acc_y, acc_z = datapoint[:3]
                gyro_x, gyro_y, gyro_z = datapoint[3:6]

                # Update circular buffer with new data
                accelerometer[current_index] = [acc_x, acc_y, acc_z]
                gyroscope[current_index] = [gyro_x, gyro_y, gyro_z]

                current_index = (current_index + 1) % max_samples

    if current_index > 9:  # Ensure enough samples for filtering
        # Calculate angle from accelerometer
        accel_angles = np.arctan2(accelerometer[:, 1], accelerometer[:, 2]) * 180 / np.pi

        # Apply low-pass filter to accelerometer angles
        accel_angles_filtered = low_pass_filter(accel_angles)

        # Apply high-pass filter to gyroscope rates
        gyro_rates = gyroscope[:, 0]  # Assuming gyro_x is the rate of change of the angle
        gyro_rates_filtered = high_pass_filter(gyro_rates)

        # Update angle using complementary filter
        for i in range(len(accel_angles_filtered)):
            angle = alpha * (angle + gyro_rates_filtered[i] * dt) + (1 - alpha) * accel_angles_filtered[i]
            angles[i] = angle

    # Clear axes
    ax1.clear()
    ax2.clear()
    ax3.clear()

    # Plot accelerometer data
    ax1.plot(accelerometer[:, 0], label='X-axis')
    ax1.plot(accelerometer[:, 1], label='Y-axis')
    ax1.plot(accelerometer[:, 2], label='Z-axis')
    ax1.set_title('Accelerometer Data')
    ax1.legend()

    # Plot gyroscope data
    ax2.plot(gyroscope[:, 0], label='X-axis')
    ax2.plot(gyroscope[:, 1], label='Y-axis')
    ax2.plot(gyroscope[:, 2], label='Z-axis')
    ax2.set_title('Gyroscope Data')
    ax2.legend()

    # Plot angle data
    ax3.plot(angles, label='Angle')
    ax3.set_title('Estimated Angle from Complementary Filter')
    ax3.legend()

# Animate the plot
ani = animation.FuncAnimation(fig, animate, interval=50)
plt.tight_layout()
plt.show()

# Stop the sensor manager with exception handling
try:
    manager.stop()
except Exception as e:
    print(f"Error stopping sensor manager: {e}")