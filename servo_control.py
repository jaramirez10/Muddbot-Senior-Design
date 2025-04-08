# Set up libraries and overall settings
from gpiozero import Servo  
from time import sleep      

servo = Servo(18)

servo.min()
sleep(1)
servo.mid()
sleep(1)
servo.max()
sleep(1)

#while True:
   # servo.min()
   # sleep(1)
   # servo.mid()
   # sleep(1)
   # servo.max()
   # sleep(1)

# Note on controlling servo:
# Value property can also be used to move servo to a particular position
# Scale is from -1 (min) to 1 (max), where 0 is the midpoint
# Min = right-most turn, Mid = straight, Max = left-most turn
