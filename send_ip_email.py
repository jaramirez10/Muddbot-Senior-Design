#!/usr/bin/env python3

import socket
import smtplib
from email.message import EmailMessage

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # connect to an external IP, doesn't actually send data
        s.connect(('8.8.8.8', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = 'Unable to determine IP'
    finally:
        s.close()
    return IP

def send_email(ip):
    msg = EmailMessage()
    msg.set_content(f'Raspberry Pi IP Address: {ip}')
    msg['Subject'] = 'Muddbot IP on Boot'
    msg['From'] = 'muddbot2025@gmail.com'
    msg['To'] = 'julioram228@gmail.com'

    # Send using Gmail (or replace with your SMTP provider)
    smtp_user = 'muddbot2025@gmail.com'
    smtp_pass = 'D@vidV@ll@ncourt2025!'  # Use an app password for Gmail

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(smtp_user, smtp_pass)
        smtp.send_message(msg)

if __name__ == "__main__":
    ip = get_ip()
    send_email(ip)

