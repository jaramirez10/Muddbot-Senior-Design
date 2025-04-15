from gpiozero import DistanceSensor
import string


right_sensor = DistanceSensor(echo=17, trigger=4)

left_sensor = DistanceSensor(echo=27, trigger=22)
# Prints distance away from the sensor in meters
while True:
    print("right" +str(right_sensor.distance) + " m")
    print("left" + str(left_sensor.distance) + " m")

# Loop for printing whether in range of the sensor
  #while True:
     # ultrasonic.wait_for_in_range()
     # print("In range")
     # ultrasonic.wait_for_out_of_range()
     # print("Out of range")
