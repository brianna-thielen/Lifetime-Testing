from Phidget22.Devices.TemperatureSensor import TemperatureSensor
from Phidget22.Devices.TemperatureSensor import ThermocoupleType

class Phidget22TemperatureSensor:
    def __init__(self, channel):
        self.ch = TemperatureSensor()
        self.ch.setChannel(channel)
        # print(f"Opening Temperature Sensor connection at channel: {channel}")

    def open_connection(self):
        self.ch.openWaitForAttachment(1000)

    def set_thermocouple_type(self, thermocouple_type):
        if thermocouple_type == "K":
            self.ch.setThermocoupleType(ThermocoupleType.THERMOCOUPLE_TYPE_K)
        elif thermocouple_type == "J":
            self.ch.setThermocoupleType(ThermocoupleType.THERMOCOUPLE_TYPE_J)
        elif thermocouple_type == "E":
            self.ch.setThermocoupleType(ThermocoupleType.THERMOCOUPLE_TYPE_E)
        elif thermocouple_type == "T":
            self.ch.setThermocoupleType(ThermocoupleType.THERMOCOUPLE_TYPE_T)
        # print(f"Setting Thermocouple Type to {thermocouple_type}...")

    def get_thermocouple_type(self):
        thermocouple_type = self.ch.getThermocoupleType()
        # print(f"Thermocouple Type: {thermocouple_type}")
        return thermocouple_type

    
    def get_temperature(self):
        temperature = self.ch.getTemperature()
        # print(f"Temperature: {temperature}")
        return temperature
    
    def close(self):
        # print("Closing Temperature Sensor connection...")
        self.ch.close()