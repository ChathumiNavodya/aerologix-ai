import os
import re

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except Exception:
    Client = None
    TWILIO_AVAILABLE = False


SANDBOX_FROM = "whatsapp:+14155238886"


def normalize_whatsapp_number(number: str) -> str:
    number = str(number or "").strip().replace(" ", "")

    if not number:
        return ""

    if number.startswith("whatsapp:+"):
        return number

    if number.startswith("+"):
        return "whatsapp:" + number

    if number.startswith("94"):
        return "whatsapp:+" + number

    if number.startswith("0") and len(number) >= 10:
        return "whatsapp:+94" + number[1:]

    return number


def is_valid_whatsapp_number(number: str) -> bool:
    number = normalize_whatsapp_number(number)
    return bool(re.fullmatch(r"whatsapp:\+\d{10,15}", number))


def clean_twilio_error(exc: Exception) -> str:
    text = re.sub(r"\x1b\[[0-9;]*m", "", str(exc)).strip()

    if "63007" in text or "specified From address" in text:
        return "Invalid WhatsApp sender. Use TWILIO_WHATSAPP_FROM=whatsapp:+14155238886"

    if "20003" in text or "Authenticate" in text:
        return "Twilio authentication failed. Check Account SID and Auth Token."

    return text


def send_whatsapp_alert(to_number: str, message: str) -> tuple[bool, str]:
    if not TWILIO_AVAILABLE:
        return False, "Twilio is not installed. Run: pip install twilio"

    sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    from_number = os.getenv("TWILIO_WHATSAPP_FROM", SANDBOX_FROM).strip()

    if from_number.startswith("+"):
        from_number = "whatsapp:" + from_number

    if from_number != SANDBOX_FROM:
        return False, f"Wrong sender. Use {SANDBOX_FROM}"

    to_number = normalize_whatsapp_number(to_number)

    if not sid or not token:
        return False, "Twilio credentials are missing."

    if not is_valid_whatsapp_number(to_number):
        return False, "Use number like +94703394005 or whatsapp:+94703394005"

    try:
        client = Client(sid, token)
        msg = client.messages.create(
            body=message,
            from_=from_number,
            to=to_number,
        )
        return True, f"WhatsApp alert sent successfully. SID: {msg.sid}"
    except Exception as exc:
        return False, "WhatsApp send failed: " + clean_twilio_error(exc)
