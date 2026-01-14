/*  Script to read data from I2C humidity sensor
 *  Sensor information: https://www.digikey.com/en/products/detail/amphenol-advanced-sensors/CC2D35S-SIP/4732682?s=N4IgTCBcDa4MwFYC0BGOA2dSByAREAugL5A
 *  Code adapted from https://howtomechatronics.com/tutorials/arduino/how-i2c-communication-works-and-how-to-use-it-with-arduino/
 *  EA: Added with 4-channel I2C multiplexer
 */

#include <Wire.h>

int SensorAddress = 0x28;  // Device address in which is also included the 8th bit for selecting the mode, read in this case.

int RH0, RH1, T0, T1, RH0_trimmed, T1_trimmed, count, mins;
float RH_calc, T_calc;
int set1, set2, set3, set4;

// Those are Enable pins for 4 different i2c bus, allowing to connect up to 4 sensors
// that have the same i2c address. Let to LOW to Disable. Set to HIGH one at a time to Enable
int i2c1_en = 5;
int i2c2_en = 2;
int i2c3_en = 3;
int i2c4_en = 4;

void setup() 
{
  Wire.begin();  // Initiate the Wire library
  Serial.begin(9600);
  count = 0;
  pinMode(i2c1_en, OUTPUT);
  pinMode(i2c2_en, OUTPUT);
  pinMode(i2c3_en, OUTPUT);
  pinMode(i2c4_en, OUTPUT);
  // Start with all 4x Enable pins LOW. Set to HIGH one at a time to read each sensor.
  digitalWrite(i2c1_en, LOW);
  digitalWrite(i2c2_en, LOW);
  digitalWrite(i2c3_en, LOW);
  digitalWrite(i2c4_en, LOW);
}

void loop() 
{
  mins = millis() / 1000 / 60;

  Serial.print("mins elapsed=");
  Serial.print(mins);

  for (int s = 1; s <= 4; s++) {
    if (s == 1) {
      set1 = HIGH; set2 = LOW; set3 = LOW; set4 = LOW;
      Serial.print(" Ambient:");
    }
    else if (s == 2) {
      set1 = LOW; set2 = HIGH; set3 = LOW; set4 = LOW;
      Serial.print(" ENCAP-C-100-1:");
    }
    else if (s == 3) {
      set1 = LOW; set2 = LOW; set3 = HIGH; set4 = LOW;
      Serial.print(" ENCAP-C-100-2:");
    }
    else if (s == 4) {
      set1 = LOW; set2 = LOW; set3 = LOW; set4 = HIGH;
      Serial.print(" ENCAP-C-25-2:");
    }

    digitalWrite(i2c1_en, set1);
    digitalWrite(i2c2_en, set2);
    digitalWrite(i2c3_en, set3);
    digitalWrite(i2c4_en, set4);

    Wire.beginTransmission(SensorAddress);
    Wire.endTransmission();

    delay(50);

    Wire.requestFrom(SensorAddress, 4);
    RH0 = Wire.read();          // Reads the data from the register
    RH1 = Wire.read();
    T0 = Wire.read();
    T1 = Wire.read();

    RH0_trimmed = RH0 & 0b111111;
    T1_trimmed = T1 & 0b11111100;
    
    RH_calc = 100 * (RH0_trimmed*256 + RH1*1.) / (16384);
    T_calc = (T0*64. + T1_trimmed*0.25) / (16384) * 165 - 40;

    Serial.print(" RH=");
    Serial.print(RH_calc);
    Serial.print(" T=");
    Serial.print(T_calc);

  }

  Serial.println();

  count = count + 1;
  delay(60000*0.25);
}