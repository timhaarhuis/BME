"""
 This Demo shows how to plot live data from the sensor and the ease of swapping between live and recorded data.
 The data can either be live from the sensor using the SensorManager,
 or it can be recorded data which is emulated like live data using the SensorEmulator.

 If you didn't already record a sensor with the RecorderDemo.py yet, do that first because that data is needed for this demo.

 Steps to run this demo:
    1. Change `USE_RECORDED_DATA` to `True` to use recorded data or `False` to use live data.
        1.1. If you want to use recorded data, make sure you have recorded data using the RecorderDemo.py.
    2. Change the `SensorNames` to the name of the sensor you want to use.
"""

# Prevent PyCharm from taking over control of the plot window backend
import os
import matplotlib
if os.name == 'posix':
    matplotlib.use("macosx")
else:
    matplotlib.use("tkagg")

# Import the necessary libraries
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import style

# Import the CreaTeBME package after installing it with: `pip install CreaTeBME`
from CreaTeBME import SensorEmulator, SensorManager

# Change this to be either using live data or emulate it using the recorded file.
USE_RECORDED_DATA = False
REPEAT_RECORDING = True
RECORDING_FILE = 'demoRecording'
SENSORS_NAMES = ['0BE6']  # Change 0FD6 to your sensor name,

# Constants:
SAMPLE_RATE = 100
STORED_SECONDS = 5
BUFFER_SIZE = SAMPLE_RATE * STORED_SECONDS
AXIS_NAMES = ["Accelerometer x", "Accelerometer y", "Accelerometer z", "Gyroscope x", "Gyroscope y", "Gyroscope z"]

# Create either a sensor manager or a sensor emulator depending if live data is needed
manager = SensorEmulator(RECORDING_FILE) if USE_RECORDED_DATA else SensorManager(SENSORS_NAMES)
manager.set_sample_rate(SAMPLE_RATE)

# Style the graph (optional)
style.use('ggplot')

# Create sub plots for accelerometer data and gyroscope.
title = f"BME IMU {'recorded' if USE_RECORDED_DATA else 'live'} data plotter demo"
fig, sub_plots = plt.subplots(len(SENSORS_NAMES), 2, num=title)
fig.set_tight_layout(True)
fig.suptitle(title, fontsize=16)

# Create a np array to store the data for each sensor
time_data = {name: np.array([0.]) for name in SENSORS_NAMES}
accelerometer_data = {sensor_name: np.array([[0, 0, 0]]) for sensor_name in SENSORS_NAMES}
gyro_data = {sensor_name: np.array([[0, 0, 0]]) for sensor_name in SENSORS_NAMES}


def update_graph(i) -> None:
    """
    UpdateGraph updates the graph with the new measurements to show the live data
    """
    global manager
    measurements = manager.get_measurements()
    for sensor, data in measurements.items():
        if len(data) == 0: continue
        for data_point in data:
            # Calculate the new time, which is the old one plus the step between each sample (one over the sample rate)
            # time_data[sensor][-BUFFER_SIZE:] retrieves the last BUFFER_SIZE (5 seconds) of timestamps from the sensor
            new_timestamp = time_data[sensor][-1] + 1.0 / float(SAMPLE_RATE)
            time_data[sensor] = np.append(time_data[sensor][-BUFFER_SIZE:], new_timestamp)

            # Add all the values of the IMU to the corresponding lists
            accelerometer_data[sensor] = np.vstack((accelerometer_data[sensor][-BUFFER_SIZE:], data_point[:3]))
            gyro_data[sensor] = np.vstack((gyro_data[sensor][-BUFFER_SIZE:], data_point[3:]))

        # If multiple sensors are connected, then select the correct row, otherwise use the sub_plots
        row_plot = sub_plots[SENSORS_NAMES.index(sensor)] if len(SENSORS_NAMES) > 1 else sub_plots

        # Update the graph to show the new measurements
        row_plot[0].clear()
        row_plot[1].clear()
        row_plot[0].plot(time_data[sensor], accelerometer_data[sensor], label=AXIS_NAMES[:3])
        row_plot[1].plot(time_data[sensor], gyro_data[sensor], label=AXIS_NAMES[3:])
        row_plot[0].legend(loc='upper right')
        row_plot[1].legend(loc='upper right')

        # Update the graph titles
        row_plot[0].set_ylabel("Acceleration in g (9.81 m/s^2)", fontsize=8)
        row_plot[0].set_title('Accelerometer', fontsize=12)
        row_plot[1].set_ylabel("Rotational velocity in degree/second", fontsize=8)
        row_plot[1].set_title('Gyroscope', fontsize=12)

    # Once the emulation is over close the graph.
    if not manager.is_running():
        print(f"Emulation ended, {'restarting..' if REPEAT_RECORDING else 'shutting down...'}")
        if REPEAT_RECORDING and USE_RECORDED_DATA:
            manager.stop()
            manager = SensorEmulator(RECORDING_FILE)
            manager.start()
        else:
            plt.close("all")


# Start the sensor manager
manager.start()

# Register the animation function that will be ran given milliseconds (in this case it depends on the sample_rate)
animatedFunction = animation.FuncAnimation(fig, update_graph, interval=1000.0 / float(SAMPLE_RATE), cache_frame_data=False)
plt.show()

# Stop the sensor manager
manager.stop()
