import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

SMTP_HOST = os.getenv("SES_SMTP_HOST")
SMTP_PORT = int(os.getenv("SES_SMTP_PORT", 587))
SMTP_USER = os.getenv("SES_SMTP_USERNAME")
SMTP_PASS = os.getenv("SES_SMTP_PASSWORD")
SENDER    = os.getenv("SES_SENDER_EMAIL")

def send_email(to: str, subject: str, body_html: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SENDER
    msg["To"]      = to
    msg.attach(MIMEText(body_html, "html"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SENDER, to, msg.as_string())