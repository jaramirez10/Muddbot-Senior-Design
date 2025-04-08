from gpiozero import DistanceSensor
import string


ultrasonic = DistanceSensor(echo=17, trigger=4)
# Prints distance away from the sensor in meters
while True:
    print(str(ultrasonic.distance) + " m")

# Loop for printing whether in range of the sensor
  #while True:
     # ultrasonic.wait_for_in_range()
     # print("In range")
     # ultrasonic.wait_for_out_of_range()
     # print("Out of range")
