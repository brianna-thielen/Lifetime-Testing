
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from equipment.phidget_4input_temperature import Phidget22TemperatureSensor as phidget

TEMP_SENSOR_DRY_BATH_CHANNEL = 2
THERMOCOUPLE_TYPE_J = 1

temperature_sensor_dry_bath = phidget(TEMP_SENSOR_DRY_BATH_CHANNEL)
temperature_sensor_dry_bath.open_connection()
temperature_sensor_dry_bath.set_thermocouple_type(THERMOCOUPLE_TYPE_J)
# temperature_sensor_dry_bath.get_thermocouple_type()
time.sleep(0.5)
temp = temperature_sensor_dry_bath.get_temperature()
time.sleep(0.5)
temperature_sensor_dry_bath.close()

print(f"Temperature: {temp}")