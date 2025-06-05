import time
import math
import matplotlib.pyplot as plt
from equipment.intan_rhs import IntanRHS as intan

WAVEFORM_BUFFER_PER_SECOND_PER_CHANNEL = 200000  # buffer size for reading from TCP waveform socket (maximum expected for 1 second)
# There is TCP lag in starting/stopping acquisition; exact number of data blocks may vary slightly
# From intan: 30 kHz, 1 channel, 1 second of wideband waveform data is 181,420 byte
NUMBER_OF_PULSES = 3
PULSE_AMPLITUDE = 1000
PULSE_WIDTH = 2000
INTERPHASE_DELAY = 1000
CHANNELS = ["a-024"]


def main():
    """Connects via TCP to RHX software, sets up stimulation parameters on a
    single channel, and runs for 5 seconds (stimulating 5 times in total).
    """

    # Connect to RHX software via TCP
    rhx = intan()
    rhx.connect_to_waveform_server()

    # Query sample rate from RHX software.
    sample_frequency = rhx.find_sample_frequency()

    # Measure impedance
    impedance_folder = f"C:/Users/3DPrint-Integral/src/integral_npsw/Intan-Control/data/impedance/"
    rhx.measure_impedance(impedance_folder)

    # Clear data output and disable all TCP channels
    rhx.reset()

    # Set up stimulation parameters on channel A-024
    rhx.set_stim_parameters("a-024", PULSE_AMPLITUDE, PULSE_WIDTH, INTERPHASE_DELAY, 5)

    # Enable data output
    rhx.enable_data_output('a-024')

    # Start board running
    rhx.start_board()

    # Stimulate
    # for i in range(NUMBER_OF_PULSES):
    #     print(f"Stimulating pulse {i + 1} of {NUMBER_OF_PULSES}")
    #     time.sleep(0.1)
    #     rhx.trigger_stim()
    time.sleep(5)

    # Stop board
    rhx.stop_board()

    # Read data from board
    buffer_size = len(CHANNELS) * WAVEFORM_BUFFER_PER_SECOND_PER_CHANNEL
    time_seconds, voltage_microvolts = rhx.read_data(buffer_size, sample_frequency)

    # plot
    plt.plot(time_seconds, voltage_microvolts)
    plt.title('A-024 Amplifier Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (uV)')
    plt.show()

    # Close TCP socket
    rhx.close_tcp()


if __name__ == '__main__':
    # Declare buffer size for reading from TCP command socket
    # This is the maximum number of bytes expected for 1 read. 1024 is plenty
    # for a single text command.
    # Increase if many return commands are expected.
    COMMAND_BUFFER_SIZE = 1024

    main()
