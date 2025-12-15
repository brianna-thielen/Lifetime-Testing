import pyvisa


class KeithleyMUX:
    """
    A class representing a Keithley Multiplexer (MUX) instrument.
    Args:
        gpib_address (str): The GPIB address of the MUX instrument.
    Attributes:
        gpib_address (str): The GPIB address of the MUX instrument.
        rm (pyvisa.ResourceManager): The resource manager for VISA communication.
        mux (pyvisa.resources.Resource): The open resource for the MUX instrument.
    Methods:
        __init__(self, gpib_address): Initializes the KeithleyMUX object.
        reset(self): Resets the MUX instrument.
        initialize(self): Initializes the MUX instrument.
        reset_and_initialize(self): Resets and initializes the MUX instrument.
        open_channels(self, bus, channels): Opens the specified channels on the MUX instrument.
        close_channels(self, bus, channels): Closes the specified channels on the MUX instrument.
        close(self): Closes the connection to the MUX instrument.
    """

    def __init__(self, gpib_address):
        """
        Initializes the Keithley MUX object.

        Args:
            gpib_address (str): The GPIB address of the MUX instrument.

        Returns:
            None
        """
        self.gpib_address = gpib_address
        self.rm = pyvisa.ResourceManager()
        self.mux = self.rm.open_resource(self.gpib_address)
        idn = self.mux.query("*IDN?")
        # print("-" * 100)
        # print(f"MUX Instrument ID: '{idn}'")
        # print("-" * 100)

    def reset(self):
        """
        Resets the MUX by sending the *RST command.

        This function resets the MUX to its default state.

        Parameters:
            None

        Returns:
            None
        """
        self.mux.write("*RST")
        # print("MUX resetting...")

    def initialize(self):
        """
        Initializes the MUX by sending the "INIT" command.

        This function is responsible for initializing the MUX before performing any operations.
        It sends the "INIT" command to the MUX and prints a message indicating that the MUX is initializing.

        Parameters:
            None

        Returns:
            None
        """
        self.mux.write("INIT")
        # print("MUX initializing...")

    def reset_and_initialize(self):
        """
        Resets and initializes the object.

        This method calls the `reset()` and `initialize()` methods to reset and initialize the object.
        """
        self.reset()
        self.initialize()

    def open_channels(self, bus, channels):
        """
        Opens the specified channels on the MUX.

        Parameters:
        - bus (str): The bus number.
        - channels (list): A list of channel numbers to be opened.

        Returns:
        None
        """
        for channel in channels:
            command = f"OPEN (@ {bus}!{channel})"
            # print("MUX: ", command)
            self.mux.write(command)

    def close_channels(self, bus, channels):
        """
        Close the specified channels on the MUX.

        Parameters:
        - bus (str): The bus number.
        - channels (list): A list of channel numbers to be closed.

        Returns:
        None
        """
        for channel in channels:
            command = f"CLOSe (@ {bus}!{channel})"
            # print("MUX: ", command)
            self.mux.write(command)

    def close(self):
        """
        Closes the MUX connection.

        If the MUX connection is open, it will be closed. Additionally, the resource manager connection will also be closed.

        Returns:
            None
        """
        if self.mux:
            self.mux.close()
        self.rm.close()
        print("Closing MUX connection...")
