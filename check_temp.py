from phidget_4input_temperature import Phidget22TemperatureSensor as phidget
import time

_TEMP_SENSOR_DRY_BATH_CHANNEL = 2
_THEMOCOUPLE_TYPE_J = 1

temperature_sensor_dry_bath = phidget(_TEMP_SENSOR_DRY_BATH_CHANNEL)
temperature_sensor_dry_bath.open_connection()
temperature_sensor_dry_bath.set_thermocouple_type(_THEMOCOUPLE_TYPE_J)
# temperature_sensor_dry_bath.get_thermocouple_type()
time.sleep(0.5)
temperature_sensor_dry_bath.get_temperature()
time.sleep(0.5)
temperature_sensor_dry_bath.close()