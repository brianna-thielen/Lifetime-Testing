import pyvisa

class KeithleySourceMeter:
    """
    Represents an sourcemeter instrument for impedance testing.
    """

    def __init__(self, gpib_address):
        """
        Initializes the sourcemeter connection.

        Args:
            address (str): The address of the sourcemeter instrument.

        Returns:
            None
        """
        self.gpib_address = gpib_address
        self.rm = pyvisa.ResourceManager()
        self.sourcemeter = self.rm.open_resource(self.gpib_address)
        idn = self.sourcemeter.query("*IDN?")
        print("-" * 100)
        print(f"SourceMeter Instrument ID: '{idn}'")
        print("-" * 100)


    def reset(self):
        """
        Resets the sourcemeter instrument.

        Returns:
            None
        """
        self.sourcemeter.write("*RST")
    

    def initialize(self):
        """
        Initializes the sourcemeter instrument.

        Returns:
            None
        """
        self.sourcemeter.write("INIT")
        print("SourceMeter initializing...")

    def reset_and_initialize(self):
        """
        Resets and initializes the sourcemeter instrument.

        Returns:
            None
        """
        self.reset()
        self.set_output("ON")
        self.initialize()
    
    def set_source(self, type: str):
        self.sourcemeter.write(f":SOUR:FUNC {type}")
        print(f"Source set to {type}")

    def set_sense(self, type: str):
        self.sourcemeter.write(f":SENS:FUNC {type}")
        print(f"Sense set to {type}")
    
    def set_current_compliance(self, value: float):
        self.sourcemeter.write(f":SENS:CURR:PROT {value}")
        print(f"Current compliance set to {value} A")

    def set_output(self, state: str):
        self.sourcemeter.write(f":OUTP {state}")
        print(f"Output {state}")

    def set_voltage(self, value: float):
        self.sourcemeter.write(f":SOUR:VOLT:LEV {value}")
        # print(f"Voltage set to {value} V")

    def measure_current(self):
        measurements = self.sourcemeter.query(":READ?").split(",")
        current = float(measurements[1])
        # print(f"Current measured: {current} A")
        return current

    def close(self):
        """
        Closes the connection to the sourcemeter instrument.

        Returns:
            None
        """
        self.sourcemeter.close()
        print("Closing SourceMeter connection...")
