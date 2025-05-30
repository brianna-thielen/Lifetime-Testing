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

    def set_voltage_compliance(self, value: float):
        self.sourcemeter.write(f":SENS:VOLT:PROT {value}")
        print(f"Voltage compliance set to {value} A")

    def set_output(self, state: str):
        self.sourcemeter.write(f":OUTP {state}")
        print(f"Output {state}")

    def set_voltage(self, value: float):
        self.sourcemeter.write(f":SOUR:VOLT:LEV {value}")
        # print(f"Voltage set to {value} V")

    def set_current(self, value: float):
        self.sourcemeter.write(f":SOUR:Curr:LEV {value}")
        print(f"Current set to {value} A")
    
    def set_4wiresens(self):
        self.sourcemeter.write(f":SYST:RSEN ON")
        # self.set_output("ON")
        print(f"4 wire sense mode on (use reference electrode)")

    def measure_current(self):
        measurements = self.sourcemeter.query(":READ?").split(",")
        current = float(measurements[1])
        # print(f"Current measured: {current} A")
        return current
    
    def measure_voltage(self):
        measurements = self.sourcemeter.query(":READ?").split(",")
        voltage = float(measurements[1])
        # print(f"Current measured: {voltage} V")
        return voltage
    
    def setup_buffer(self, points: int, delay: float):
        self.sourcemeter.write(f":TRAC:FEED SENS")
        self.sourcemeter.write(f":TRAC:POIN {points}")
        self.sourcemeter.write(f":TRAC:FEED:CONT NEXT")
        self.sourcemeter.write(f":TRIG:COUN {points}")
        self.sourcemeter.write(f":TRIG:DEL {delay}")
        print(f"Buffer enabled")

    def start_buffer(self):
        self.sourcemeter.write(f":INIT")
        print(f"Buffer started")

    def read_buffer(self):
        data = self.sourcemeter.query(f":TRAC:DATA?")
        print(f"Buffer read")
        return data

    def clear_buffer(self):
        self.sourcemeter.write(f":TRAC:CLE")
        print(f"Buffer cleared")

    def close(self):
        """
        Closes the connection to the sourcemeter instrument.

        Returns:
            None
        """
        self.set_output("OFF")
        self.sourcemeter.close()
        print("Closing SourceMeter connection...")
