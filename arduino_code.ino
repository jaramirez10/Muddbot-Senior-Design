#include <AFMotor.h>
String msg;

AF_DCMotor motor(3);

void setup() 
{
  Serial.begin(9600);
  motor.setSpeed(100);
  motor.run(RELEASE);
}

void loop() 
{
  uint8_t i;
  readSerialPort();
  if (msg=="FORWARD") {
    motor.run(FORWARD);
    }
  else if (msg =="BACKWARD") {
    motor.run(BACKWARD);
    }
  else if (msg == "STOP") {
    motor.run(RELEASE);
    }
  else if (msg.substring(0,5) == "SPEED") {
    i = atoi(msg.substring(6))
    sendData(msg.subtstring(6))
    motor.setSpeed(i);
    }
  else {
    if (msg != ""){
      sendData("couldnt parse command")
    }
    }
}

void readSerialPort() {
  msg = "";
  if (Serial.available()) {
    delay(10);
    while (Serial.available() > 0) {
      msg += (char)Serial.read();
    }
    Serial.flush();
  }
}


void sendData(string) {
  //write data
  Serial.print(nom);
  Serial.print(" : ");
  Serial.print(string);
}
