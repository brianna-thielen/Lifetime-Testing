"""
Intan RHS control module for Python
Based off sample code from https://intantech.com/files/ RHX_TCP.zip
"""

import time
import datetime
import os
import socket

COMMAND_BUFFER_SIZE = 1024

class IntanRHS:
    """
    Represents an Intan RHS stimulator for current stimulation and voltage recording.
    """

    def __init__(self):
        """
        Initializes the intan connection.

        Args:
            address (str): The address of the sourcemeter instrument.

        Returns:
            None
        """

        # Connect to TCP command server - default home IP address at port 5000.
        print('Connecting to TCP command server...')
        self.scommand = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.scommand.connect(('127.0.0.1', 5000))

        # Query runmode from RHX software
        self.scommand.sendall(b'get runmode')
        commandReturn = str(self.scommand.recv(COMMAND_BUFFER_SIZE), "utf-8")
        isStopped = commandReturn == "Return: RunMode Stop"

        # If controller is running, stop it
        if not isStopped:
            self.scommand.sendall(b'set runmode stop')
            time.sleep(0.1)

    def connect_to_waveform_server(self):
        # Connect to TCP waveform server - default home IP address at port 5001.
        print('Connecting to TCP waveform server...')
        self.swaveform = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.swaveform.connect(('127.0.0.1', 5001))

    def disconnect_from_waveform_server(self):
        print('Disconnecting from TCP waveform server...')
        self.swaveform.close()
    
    def reset(self):
        """
        Resets the intan instrument.
        """
        # Ensure the instrument is stopped
        self.scommand.sendall(b'set runmode stop')
        time.sleep(0.1)

        # Clear TCP data output to ensure no TCP channels are enabled.
        self.scommand.sendall(b'execute clearalldataoutputs')
        time.sleep(0.1)

    def set_stim_parameters(self, channel, amplitude, pulsewidth, interphase, frequency, name):
        """
        Programs settings for a single current pulse or pulse train.

        Inputs:
            channel (str): The channel to stimulate (e.g. 'a-010').
            amplitude (int): The amplitude of the current pulse in microamps.
            duration (int): The duration of the current pulse in microseconds.
            interphase (int): The interphase delay in microseconds.
            frequency (float): Frequency of stimulation pulse train in Hz. 0 for single pulse.
        """
        # Enable stim on channel
        # self.scommand.sendall(b'set %s.stimenabled true' % channel)
        self.scommand.sendall(f'set {channel}.stimenabled true'.encode('utf-8'))
        time.sleep(0.1)

        # Set trigger to keypress F1 (this is used to trigger stim later)
        self.scommand.sendall(f'set {channel}.source keypressf1'.encode('utf-8'))
        time.sleep(0.1)

        # For continuous pulsing, set to level triggered on low 
        # This allows continuous pulsing at a frequency set by post stim refractory period
        if frequency != 0:
            refractory_period = (1 / frequency * 1000000) - 2*pulsewidth - interphase
            self.scommand.sendall(f'set {channel}.triggeredgeorlevel level'.encode('utf-8'))
            time.sleep(0.1)
            self.scommand.sendall(f'set {channel}.triggerhighorlow low'.encode('utf-8'))
            time.sleep(0.1)
            self.scommand.sendall(f'set {channel}.refractoryperiodmicroseconds {refractory_period}'.encode('utf-8'))
            time.sleep(0.1)
        else:
            self.scommand.sendall(f'set {channel}.triggeredgeorlevel edge'.encode('utf-8'))
            time.sleep(0.1)
            self.scommand.sendall(f'set {channel}.triggerhighorlow high'.encode('utf-8'))
            time.sleep(0.1)
            self.scommand.sendall(f'set {channel}.refractoryperiodmicroseconds 0'.encode('utf-8'))
            time.sleep(0.1)

        # Set stim shape to biphasic with delay, negative first
        self.scommand.sendall(f'set {channel}.shape biphasicwithinterphasedelay'.encode('utf-8'))
        time.sleep(0.1)
        self.scommand.sendall(f'set {channel}.polarity negativefirst'.encode('utf-8'))
        time.sleep(0.1)

        # Set stim parameters
        self.scommand.sendall(f'set {channel}.firstphaseamplitudemicroamps {amplitude}'.encode('utf-8'))
        time.sleep(0.1)
        self.scommand.sendall(f'set {channel}.secondphaseamplitudemicroamps {amplitude}'.encode('utf-8'))
        time.sleep(0.1)
        self.scommand.sendall(f'set {channel}.firstphasedurationmicroseconds {pulsewidth}'.encode('utf-8'))
        time.sleep(0.1)
        self.scommand.sendall(f'set {channel}.secondphasedurationmicroseconds {pulsewidth}'.encode('utf-8'))
        time.sleep(0.1)
        self.scommand.sendall(f'set {channel}.interphasedelaymicroseconds {interphase}'.encode('utf-8'))
        time.sleep(0.1)

        # Disable amp settle
        self.scommand.sendall(f'set {channel}.enableampsettle false'.encode('utf-8'))
        time.sleep(0.1)

        # Enable charge recovery
        self.scommand.sendall(f'set {channel}.enablechargerecovery true'.encode('utf-8'))
        time.sleep(0.1)
        self.scommand.sendall(f'set {channel}.poststimchargerecovonmicroseconds 0'.encode('utf-8'))
        time.sleep(0.1)
        self.scommand.sendall(f'set {channel}.poststimchargerecovoffmicroseconds 2000'.encode('utf-8'))
        time.sleep(0.1)

        # Execute upload
        self.scommand.sendall(f'execute uploadstimparameters {channel}'.encode('utf-8'))
        time.sleep(1)

        # Set custom name
        self.scommand.sendall(f'set {channel}.customchannelname {name}'.encode('utf-8'))
        time.sleep(0.1)

    def disable_stim(self, channel):
        """
        Disables stim for a single channel
        """
        # Disable stim on channel
        self.scommand.sendall(f'set {channel}.stimenabled false'.encode('utf-8'))
        time.sleep(0.1)

    def start_board(self):
        # Send command to RHX software to start recording/stimulation
        self.scommand.sendall(b'set runmode run')
        time.sleep(0.1)

    def stop_board(self):
        # Send command to RHX software to stop recording/stimulation
        self.scommand.sendall(b'set runmode stop')
        time.sleep(0.1)
    
    def close_tcp(self):
        # Close TCP socket
        self.scommand.close()

    def trigger_stim(self):
        # Trigger a manual stim pulse
        self.scommand.sendall(b'execute manualstimtriggerpulse f1')

    def find_sample_frequency(self):
        # Query sample rate from RHX software.
        self.scommand.sendall(b'get sampleratehertz')
        commandReturn = str(self.scommand.recv(COMMAND_BUFFER_SIZE), "utf-8")
        expectedReturnString = "Return: SampleRateHertz "
        # Look for "Return: SampleRateHertz N" where N is the sample rate.
        sample_frequency = float(commandReturn[len(expectedReturnString):])
        print(f"Sample frequency: {int(sample_frequency / 1000)} kHz")
        return sample_frequency
    
    def enable_data_output(self, channel):
        # Send TCP commands to set up TCP Data Output Enabled for wide band of channel.
        self.scommand.sendall(f'set {channel}.tcpdataoutputenableddc true'.encode('utf-8'))
        time.sleep(0.1)
    
    def disable_data_output(self, channel):
        # Send TCP commands to set up TCP Data Output Enabled for wide band of channel.
        self.scommand.sendall(f'set {channel}.tcpdataoutputenableddc false'.encode('utf-8'))
        time.sleep(0.1)
    
    def read_data(self, buffer_size, sample_frequency):
        # 128 frames per block; standard data block size used by Intan
        frames_per_block = 128

        # waveformBytesPerFrame = SizeOfTimestamp + SizeOfSample;
        # timestamp is a 4-byte int, and amplifier sample is a 2-byte unsigned int
        waveform_bytes_per_frame = 4 + 2

        # SizeOfMagicNumber = 4; Magic number is a 4-byte (32-bit) unsigned int
        magic_number_bytes = 4

        waveform_bytes_per_block = frames_per_block * waveform_bytes_per_frame + magic_number_bytes
        
        # Read waveform data
        # print("Reading waveform data...")
        rawData = self.swaveform.recv(buffer_size)

        numBlocks = int(len(rawData) / waveform_bytes_per_block)

        # Index used to read the raw data that came in through the TCP socket.
        rawIndex = 0

        # List used to contain scaled timestamp values in seconds.
        amplifierTimestamps = []

        # List used to contain scaled amplifier data in microVolts.
        amplifierData = []

        # Calculate timestep from sample rate.
        timestep = 1 / sample_frequency

        for i in range(numBlocks):
            # Expect 4 bytes to be TCP Magic Number as uint32.
            # If not what's expected, raise an exception.
            magicNumber, rawIndex = readUint32(rawData, rawIndex)
            if magicNumber != 0x2ef07a08:
                raise InvalidMagicNumber('Error... magic number incorrect')

            # Each block should contain 128 frames of data - process each
            # of these one-by-one
            for _ in range(frames_per_block):
                # Expect 4 bytes to be timestamp as int32.
                rawTimestamp, rawIndex = readInt32(rawData, rawIndex)

                # Multiply by 'timestep' to convert timestamp to seconds
                amplifierTimestamps.append(rawTimestamp * timestep)

                # Expect 2 bytes of wideband data.
                rawSample, rawIndex = readUint16(rawData, rawIndex)

                # Scale this sample to convert to microVolts
                amplifierData.append(0.195 * (rawSample - 32768))

        return amplifierTimestamps, amplifierData
    
    def flush_buffer(self, buffer_size):
        print("start")
        rawData = self.swaveform.recv(buffer_size)
        print("end")
        
        
    def measure_impedance(self, folder):
        # Set file parameters
        # Create directory if it doesn't exist
        if os.path.exists(folder) == False:
            os.makedirs(folder)
            
        filename = f"intanimpedance_1k_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        # file_path = f"{folder}/{filename}.csv"

        self.scommand.sendall(f'set impedancefilename.basefilename {filename}.csv'.encode('utf-8'))
        time.sleep(1)
        self.scommand.sendall(f'set impedancefilename.path {folder}'.encode('utf-8'))
        time.sleep(1)

        # Set measurement frequency to 1 kHz.
        self.scommand.sendall(b'set desiredimpedancefreqhertz 1000')
        time.sleep(1)

        # Send TCP commands to measure impedance.
        self.scommand.sendall(b'execute measureimpedance')
        time.sleep(4)

        # Save impedance data        
        self.scommand.sendall(b'execute saveimpedance')
        time.sleep(1)

        return filename
    

def readUint32(array, arrayIndex):
    """Reads 4 bytes from array as unsigned 32-bit integer.
    """
    variableBytes = array[arrayIndex: arrayIndex + 4]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=False)
    arrayIndex = arrayIndex + 4
    return variable, arrayIndex


def readInt32(array, arrayIndex):
    """Reads 4 bytes from array as signed 32-bit integer.
    """
    variableBytes = array[arrayIndex: arrayIndex + 4]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=True)
    arrayIndex = arrayIndex + 4
    return variable, arrayIndex


def readUint16(array, arrayIndex):
    """Reads 2 bytes from array as unsigned 16-bit integer.
    """
    variableBytes = array[arrayIndex: arrayIndex + 2]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=False)
    arrayIndex = arrayIndex + 2
    return variable, arrayIndex

class InvalidMagicNumber(Exception):
    """Exception returned when the first 4 bytes of a data block are not the
    expected RHX TCP magic number (0x2ef07a08).
    """
