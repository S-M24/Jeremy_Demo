import time
import json
import serial
import sys
import time
import csv
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


laserPORT = "COM13"  

ser = serial.Serial(laserPORT, 115200, timeout=1)

def send_command(command):
    ser.write((command + "\n").encode())

def send_command_with_response(command):
    """Sends a command to the serial port and prints the response."""
    ser.write((command + "\n").encode())
    time.sleep(1.5)  # Wait for the device to process the command
    response = ser.readline().decode("ascii").strip()
    time.sleep(1.5)
    print(f"Command: {command}, Response: {response}")
    return response

def laser_setup():
        send_command("OUTPUT RS422_ASCII")
        read_serial_line(ser)
        send_command("LASER MEASURE")  
        time.sleep(0.1)
        read_serial_line(ser)
        send_command("TRIGGER SOFTWARE") 
        read_serial_line(ser)
        time.sleep(0.1) 
        send_command("TRIGGERAT OUTPUT")  
        read_serial_line(ser)
        time.sleep(0.1)
        send_command("TRIGGERLEVEL HIGH")
        read_serial_line(ser)
        time.sleep(0.1)
        send_command("TRIGGERCOUNT 1")
        read_serial_line(ser)
        time.sleep(0.1)
        send_command("MEASSETTINGS STORE")
        read_serial_line(ser)
        send_command("MEASSETTINGS READ")

def clear_serial_buffers(ser):
    """Clears both input and output buffers of the serial port."""
    ser.flushInput()  # Clear the input buffer
    ser.flushOutput() # Clear the output buffer

def read_serial_line(ser):
    try:
        # Attempt to read a line and decode it as ASCII
        line = ser.readline().decode('ascii').strip()
        print(line)
    except UnicodeDecodeError as e:
        # Handle bytes that can't be decoded with ASCII
        print(f"UnicodeDecodeError: {e}")
        # Optionally read the raw data to see what was sent
        raw_data = ser.read(ser.in_waiting)  # Read all data in the buffer
        print(f"Raw data: {raw_data}")



JSON_FILE_IDs = "detected_devices.json"
mirror_definitions = {
    "M1": "Board: BPAA0626, Device: AUAG3857",
    "M2": "Board: BPAA0627, Device: AUAG3902",
    "M3": "Board: BPAA0757, Device: AUAH1167",
    "M4": "Board: BPAA0743, Device: AUAH1175",
    "M5": "Board: BPAA0749, Device: AUAG3868",
    "M6": "Board: BPAA0622, Device: AUAG3882",
    #### ADD extra mirror here
}


def establish_serial_connection(port, baudrate=256000, timeout=1):
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        return ser
    except Exception as e:
        print(f"Error establishing serial connection on {port}: {e}")
        return None


def load_from_json(filename):
    """Load data from a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


class Mirror:
    def __init__(self, mirror_id, channel_x=None, channel_y=None):
        self.id = mirror_id
        self.port_name = None  # This stores the port name like 'COM5'
        self.port = None  # This stores the serial.Serial object
        self.channel_x = channel_x
        self.channel_y = channel_y
        self.angle_x = 0.0
        self.angle_y = 0.0

    def initialize_to_zero(self):
        self.set_angle_x(0.0)
        self.set_angle_y(0.0)

    def set_port(self, port_name):
        """Establish a serial connection and store it."""
        self.port_name = port_name  # Store the port name if needed
        self.port = establish_serial_connection(port_name)  # This should return a serial.Serial object

    def set_channel_x(self, channel):
        self.channel_x = channel

    def set_channel_y(self, channel):
        self.channel_y = channel

    def set_angle_x(self, angle):
        """Set X angle using serial communication."""
        self.angle_x = angle
        command = f"x= {angle:.4f}\r\n"
        if self.port:
            self.port.write(command.encode())

    def set_angle_y(self, angle):
        """Set Y angle using serial communication."""
        self.angle_y = angle
        command = f"y= {angle:.4f}\r\n"
        if self.port:
            self.port.write(command.encode())

    def get_angle_x(self):
        return self.angle_x

    def get_angle_y(self):
        return self.angle_y
    
    def update_angles(self, angle_x, angle_y):
        self.set_angle_x(angle_x)
        self.set_angle_y(angle_y)


def assign_ports_to_mirrors():
    try:
        detected_devices = load_from_json(JSON_FILE_IDs)
        if not detected_devices:
            print("Warning: No devices were detected from the JSON file.")
            return

        print(f"Detected Devices: {detected_devices}")

        for mirror_id, mirror_device in mirror_definitions.items():
            found = False

            for device, port_name in detected_devices.items():
                if device == mirror_device:
                    # Establish a serial connection and assign it to the mirror
                    mirrors[mirror_id].set_port(port_name)
                    print(f"Successfully connected to {port_name} for {mirror_id}.")
                    found = True
                    break

            if not found:
                print(f"Warning: No matching detected device found for {mirror_id} ({mirror_device}).")

    except Exception as e:
        print(f"Error in assign_ports_to_mirrors: {e}")


# Load saved positions from the JSON file
def load_saved_positions(filename='saved_positions.json'):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filename}.")
        return None


# Apply positions to mirrors
def apply_positions(positions, mirrors):
    for mirror_id, coords in positions.items():
        if mirror_id in mirrors:
            mirror = mirrors[mirror_id]
            mirror.set_angle_x(coords['x'])
            mirror.set_angle_y(coords['y'])
           # print(f"Set {mirror_id} to x={coords['x']}, y={coords['y']}")
        else:
            print(f"Warning: {mirror_id} not found among connected mirrors.")



def cycle_positions(mirrors, num_cycles):
    saved_positions = load_saved_positions()
    if saved_positions is None:
        return

    sequence = ["1", "2", "3", "4", "5", "6", "21", "22", "27", "31", "32", "37", "41", "42","47", "51", "52", "57", "61", "62", "67"]
    measurements = {vector: [] for vector in sequence}

    for cycle in range(num_cycles):
        for set_name in sequence:
            if set_name in saved_positions:
                print(f"Applying {set_name} in cycle {cycle + 1}...")
                positions = saved_positions[set_name]
                apply_positions(positions, mirrors)

                time.sleep(3)
                send_command("TRIGGERSW")
                time.sleep(0.5)
                ser.readline()  # Consume the 'TRIGGERSW OK' line

                try:
                    distance_line = ser.readline().decode("ascii").strip()
                    if ':' in distance_line:
                        distance = float(distance_line.split(':')[-1].strip())
                    else:
                        distance = None
                except UnicodeDecodeError as e:
                    print(f"Decode error: {e}. Attempting to read raw data...")
                    raw_data = ser.readline()
                    print(f"Raw data: {raw_data}")
                    distance = None

                if distance is not None:
                    measurements[set_name].append(distance)

                clear_serial_buffers(ser)
            else:
                print(f"Warning: Set {set_name} not found in saved positions.")

    # Calculate mean values for each position
    mean_values = {set_name: sum(distances) / len(distances) if distances else None 
                   for set_name, distances in measurements.items()}

    # Optionally,  print or return the mean values
    print("Mean Values:", mean_values)
    return mean_values

#Known Mirror coordinates 
M2 = np.array([ 0.593086809752998 , 2.34303234612275 , 0.6201752402910742 ])
M3 = np.array([ 1.163103984993054 , 2.170864235721563 , 0.4307522282848982 ])
M4 = np.array([ 1.160802617523118 , 2.169122487480087 , 0.061656943829155375 ])
M5 = np.array([ -0.022383522058118648 , 2.1670475545985965 , 0.058316318034148114 ])
M6 = np.array([ -0.012369897233757539 , 2.1698471627261724 , 0.43256901183062707 ])

# Target position guesses
T1_guess = np.array([0.2, 0.2, 0.2])
T2_guess = np.array([0.22, 0.22, 0.21])
T3_guess = np.array([0.22, 0.22, 0.21])

def distance(X, M, d):
    if len(X) != 3 or len(M) != 3:
        raise ValueError( "X and M must be tuples of length 3") 
        
    if not all(isinstance(coord, (int, float)) for coord in X) or not all(isinstance(coord, (int, float)) for coord in M):
                        raise ValueError("M and T must contain numbers")
    
    return ((X[0] - M[0])**2 + (X[1] - M[1])**2 + (X[2] - M[2])**2 - d**2)
 
#function to calculate the error between distance measured and distance calculated from guess coordinates 
def error(X_guess, d_M2X, d_M3X, d_M4X, d_M5X, d_M6X):
    return [
        distance(X_guess, M2, d_M2X),
        distance(X_guess, M3, d_M3X),
        distance(X_guess, M4, d_M4X),
        distance(X_guess, M5, d_M5X),
        distance(X_guess, M6, d_M6X),
    ]

def draw_line(p1, p2, color):
    x_values = [p1[0], p2[0]]
    y_values = [p1[1], p2[1]]
    z_values = [p1[2], p2[2]]
    ax.plot(x_values, y_values, z_values, color=color)

if __name__ == "__main__":
    # Initialize mirrors
    mirrors = {mirror_id: Mirror(mirror_id) for mirror_id in mirror_definitions}

    # Assign ports to mirrors based on detected devices
    assign_ports_to_mirrors()

    # Initialize mirrors to zero after establishing a connection
    for mirror_id, mirror in mirrors.items():
        if mirror.port is not None:  # Check if the serial connection was successful
            mirror.initialize_to_zero()  # Initialize angles to 0 after connection
        else:
            print(f"Failed to initialize {mirror_id} due to connection error.")

    # Start the cycle with a x-second delay between positions
    clear_serial_buffers(ser)
    send_command("RESET")
    time.sleep(2)  # Wait for the sensor to reboot
    laser_setup()
    time.sleep(0.5)
    clear_serial_buffers(ser)
    time.sleep(1)
    # Example usage
mean_values = cycle_positions(mirrors, 10)

M2T1 = mean_values["21"] - mean_values["2"]
M3T1 = mean_values["31"] - mean_values["3"]
M4T1 = mean_values["41"] - mean_values["4"]
M5T1 = mean_values["51"] - mean_values["5"]
M6T1 = mean_values["61"] - mean_values["6"]

M2T2 = mean_values["22"] - mean_values["2"]
M3T2 = mean_values["32"] - mean_values["3"]
M4T2 = mean_values["42"] - mean_values["4"]
M5T2 = mean_values["52"] - mean_values["5"]
M6T2 = mean_values["62"] - mean_values["6"]

M2T3 = mean_values["27"] - mean_values["2"]
M3T3 = mean_values["37"] - mean_values["3"]
M4T3 = mean_values["47"] - mean_values["4"]
M5T3 = mean_values["57"] - mean_values["5"]
M6T3 = mean_values["67"] - mean_values["6"]

#solve retroreflector position
T1 = least_squares(error, T1_guess, args=(M2T1, M3T1, M4T1, M5T1, M6T1)).x
T2 = least_squares(error, T2_guess, args=(M2T2, M3T2, M4T2, M5T2, M6T2)).x
T3 = least_squares(error, T3_guess, args=(M2T3, M3T3, M4T3, M5T3, M6T3)).x

print("T1 COORDINATES:", T1)
print("T2 COORDINATES:", T2)
print("T3 COORDINATES:", T3)
# Plotting the points and distances
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(*T1, color='blue', label='T1')
ax.scatter(*T2, color='red', label='T2')
ax.scatter(*T3, color='yellow', label='T3')
ax.scatter(*M2, color='grey', label='M2')
ax.scatter(*M3, color='green', label='M3')
ax.scatter(*M4, color='purple', label='M4')
ax.scatter(*M5, color='orange', label='M5')
ax.scatter(*M6, color='pink', label='M6')

draw_line(M2, T1, 'red')
draw_line(M2, T2, 'red')
draw_line(M2, T3, 'red')

draw_line(M3, T1, 'green')
draw_line(M3, T2, 'green')
draw_line(M3, T3, 'green')

draw_line(M4, T1, 'purple')
draw_line(M4, T2, 'purple')
draw_line(M4, T3, 'purple')

draw_line(M5, T1, 'orange')
draw_line(M5, T2, 'orange')
draw_line(M5, T3, 'orange')

draw_line(M6, T1, 'pink')
draw_line(M6, T2, 'pink')
draw_line(M6, T3, 'pink')

#labels and title
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Plot of Mirrors and Targets with Distances')

ax.legend()
plt.show()