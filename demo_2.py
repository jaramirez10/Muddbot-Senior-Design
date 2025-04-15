#!/usr/bin/env python
# -*- coding: utf-8 -*-
# lsusb to check device name
#dmesg | grep "tty" to find port name

from gpiozero import Servo
servo = Servo(18)


import serial,time
if __name__ == '__main__':
    
    print('Running. Press CTRL-C to exit.')
    with serial.Serial("/dev/ttyACM0", 9600, timeout=1) as arduino:
        time.sleep(0.1) #wait for serial to open
        if arduino.isOpen():
            print("{} connected!".format(arduino.port))
            try:
                while True:
                    cmd=input("Enter command : ")
                    if cmd[:5] == "STEER":
                        print(f"Steering to {cmd[6:]}!")
                        servo.value = float(cmd[6:])
                    else:
                        arduino.write(cmd.encode())
                        #time.sleep(0.1) #wait for arduino to answer
                        print(f"Sent command:_{cmd}_")
                        while arduino.inWaiting()==0: pass
                        if  arduino.inWaiting()>0: 
                            answer=arduino.readline()
                            print(answer)
                            arduino.flushInput() #remove data after reading
            except KeyboardInterrupt:
                print("KeyboardInterrupt has been caught.")

                