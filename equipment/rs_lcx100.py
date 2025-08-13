from RsInstrument import *  # type: ignore


class LCX100:
    """
    Represents an LCX100 instrument for impedance testing.

    Args:
        address (str): The address of the LCX100 instrument.

    Attributes:
        lcx (RsInstrument): The connection to the LCX100 instrument.

    Methods:
        reset(): Resets the LCX100 instrument.
        initialize(): Initializes the LCX100 instrument.
        set_visa_timeout(timeout): Sets the VISA timeout of the LCX100 instrument.
        set_measurement_type(measurement_type): Sets the measurement type of the LCX100 instrument.
        reset_and_initialize(): Resets and initializes the LCX100 instrument.
        set_frequency(frequency): Sets the frequency of the LCX100 instrument.
        measure_impedance(): Measures the impedance using the LCX100 instrument.
        close(): Closes the connection to the LCX100 instrument.
    """

    def __init__(self, address):
        """
        Initializes the LCX100 connection.

        Args:
            address (str): The address of the LCX100 instrument.

        Returns:
            None
        """
        self.lcx = RsInstrument(address)
        idn = self.lcx.query_str("*IDN?")
        # print("-" * 100)
        # print(f"RsInstrument driver version: {self.lcx.driver_version}")
        # print(f"Visa manufacturer: {self.lcx.visa_manufacturer}")
        # print(f"LCR Instrument ID: '{idn}'")
        # print(f"LCR Instrument full name: {self.lcx.full_instrument_model_name}")
        # print(
        #     f'LCR Instrument installed options: {",".join(self.lcx.instrument_options)}'
        # )
        # print(f"LCR Instrument Connection Status: {self.lcx.is_connection_active()}")
        # print("-" * 100)

    def reset(self):
        """
        Resets the LCX100 instrument.

        Returns:
            None
        """
        self.lcx.write("*RST")
        # print("LCR resetting...")

    def initialize(self):
        """
        Initializes the LCX100 instrument.

        Returns:
            None
        """
        self.lcx.write("INIT")
        # print("LCR initializing...")

    def set_aperture(self, aperture):
        """
        Sets the aperture of the LCX100 instrument.

        Args:
            aperture (int): The aperture value to set.

        Returns:
            None
        """
        self.lcx.write(f"APER {aperture}")
        # print(f"Setting LCR aperture to {aperture}...")

    def set_visa_timeout(self, timeout):
        """
        Sets the VISA timeout of the LCX100 instrument.

        Args:
            timeout (int): The VISA timeout value in milliseconds.

        Returns:
            None
        """
        self.lcx.visa_timeout = timeout
        # print(f"Setting LCR VISA timeout to {timeout} ms...")

    def set_measurement_type(self, measurement_type):
        """
        Sets the measurement type of the LCX100 instrument.

        Args:
            measurement_type (str): The measurement type to set.

        Returns:
            None
        """
        self.lcx.write(f"FUNC:MEAS:TYPE {measurement_type}")
        # print(f"Setting LCR measurement type to {measurement_type}...")

    def set_measurement_range(self, measurement_range):
        """
        Sets the measurement range of the LCX100 instrument.

        Args:
            measurement_range (float): The measurement range to set.

        Returns:
            None
        """
        self.lcx.write(f"FUNC:IMP:RANG {measurement_range}")
        # print(f"Setting LCR measurement range to {measurement_range}...")

    def reset_and_initialize(self):
        """
        Resets and initializes the LCX100 instrument.

        Returns:
            None
        """
        self.reset()
        self.initialize()

    def set_voltage(self, voltage):
        """
        Sets the voltage of the LCX100 instrument.

        Args:
            voltage (float): The voltage to set in V.

        Returns:
            None
        """
        self.lcx.write(f"VOLT {voltage}")
        # print(f"Setting LCR voltage to {voltage} V")

    def set_frequency(self, frequency):
        """
        Sets the frequency of the LCX100 instrument.

        Args:
            frequency (float): The frequency to set in Hz.

        Returns:
            None
        """
        self.lcx.write(f"FREQ {frequency} Hz")
        # print(f"Setting LCR frequency to {frequency} Hz")

    def get_impedance(self):
        """
        Measures the impedance using the LCX100 instrument.

        Returns:
            str: The impedance measurement result.
        """
        response = self.lcx.query("READ:IMP?")
        impedance, phase_angle = response.split(",")
        # print(f"Impedance: {impedance} Ohms, Phase Angle: {phase_angle} degrees")
        return impedance, phase_angle

    def close(self):
        """
        Closes the connection to the LCX100 instrument.

        Returns:
            None
        """
        self.lcx.close()
        print("Closing LCR connection...")
