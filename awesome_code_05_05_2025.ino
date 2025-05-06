#include <Adafruit_MotorShield.h>
#include <Servo.h>
#include <Ultrasonic.h>
// Select which 'port' M1, M2, M3 or M4. In this case, M1
Adafruit_MotorShield AFMS = Adafruit_MotorShield();
Adafruit_DCMotor *myMotor = AFMS.getMotor(3);
Servo myServo;
Ultrasonic left(6,7);
Ultrasonic right(13,12);
Ultrasonic front(5,4);

String msg;
int left_dist, right_dist, fwd_dist;

void setup() 
{
  Serial.begin(115200);
  if(!AFMS.begin(2000)) {
    Serial.println("Could not find Motor Shield. Check wiring and end script.");
    while (1);
    }
  myMotor->setSpeed(100);
  myMotor->run(RELEASE);
  myServo.attach(10);
  delay(10);
}
void loop() 
{
  uint8_t i;
  readSerialPort();
  if (msg.substring(0,7) == "FORWARD") {
    myMotor->run(FORWARD);
    sendEmptyMsg();
    }
  else if (msg.substring(0,8) == "BACKWARD") {
    myMotor->run(BACKWARD);
    sendEmptyMsg();
    }
  else if (msg.substring(0,4) == "STOP") {
    myMotor->run(RELEASE);
    sendEmptyMsg();
    }
  else if (msg.substring(0,5) == "SPEED") {
    i = (msg.substring(6)).toInt();
    myMotor->setSpeed(i);
    sendEmptyMsg();
    }
  else if (msg.substring(0,5) == "STEER") {
    i = (msg.substring(6)).toInt();
    myServo.write(i);
    sendEmptyMsg();
    }
  else if (msg.substring(0,6) == "SENSOR") {
    left_dist = left.read();
    right_dist = right.read();
    fwd_dist = front.read();
    sendDists(left_dist, fwd_dist, right_dist);
   }
}

void readSerialPort() {
  msg = "";
  if (Serial.available()) {
    delay(10);
    while (Serial.available() > 0) {
      msg += (char)Serial.read();
    }
  }
}

void sendEmptyMsg() {
  Serial.print("[]\n");
  }
void sendDists(int left_dist, int fwd_dist, int right_dist) {
  //write data
  Serial.print("[");
  Serial.print(left_dist);
  Serial.print(" / ");
  Serial.print(fwd_dist);
  Serial.print(" \\ ");
  Serial.print(right_dist);
  Serial.print("]\n");
}
