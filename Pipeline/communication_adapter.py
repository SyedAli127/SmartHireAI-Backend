import logging
import os
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

logger = logging.getLogger(__name__)


def _smtp_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


# ── Plain-SMTP provider (non-SES) ─────────────────────────────────────────────

def _send_via_smtp(*, candidate_email: str, subject: str, body: str) -> bool:
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    from_email = os.getenv("RECRUITER_FROM_EMAIL") or user or "noreply@example.com"
    use_tls = _smtp_bool("SMTP_USE_TLS", True)
    use_ssl = _smtp_bool("SMTP_USE_SSL", False)

    if not host:
        logger.warning("SMTP send failed: SMTP_HOST is not configured")
        return False

    if user and not password:
        logger.warning("SMTP send failed: SMTP_USER is set but SMTP_PASS is missing")
        return False

    msg = EmailMessage()
    msg.set_content(body)
    msg["To"] = candidate_email
    msg["Subject"] = subject
    msg["From"] = from_email

    try:
        if use_ssl:
            with smtplib.SMTP_SSL(host, port, timeout=30) as server:
                if user:
                    server.login(user, password or "")
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=30) as server:
                if use_tls:
                    server.starttls()
                if user:
                    server.login(user, password or "")
                server.send_message(msg)
        return True
    except Exception as e:
        logger.warning("SMTP send failed for %s: %s", candidate_email, e)
        return False


# ── Amazon SES SMTP provider ──────────────────────────────────────────────────

def _send_via_ses(*, candidate_email: str, subject: str, body_text: str, body_html: str) -> bool:
    host     = os.getenv("SES_SMTP_HOST")
    port     = int(os.getenv("SES_SMTP_PORT", "587"))
    user     = os.getenv("SES_SMTP_USERNAME")
    password = os.getenv("SES_SMTP_PASSWORD")
    sender   = os.getenv("SES_SENDER_EMAIL") or user or "noreply@example.com"

    if not host:
        logger.warning("SES send failed: SES_SMTP_HOST is not configured")
        return False
    if not user or not password:
        logger.warning("SES send failed: SES_SMTP_USERNAME / SES_SMTP_PASSWORD not configured")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = candidate_email
    msg.attach(MIMEText(body_text, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(sender, candidate_email, msg.as_string())
        logger.info("SES email sent to %s | %s", candidate_email, subject)
        return True
    except Exception as e:
        logger.warning("SES send failed for %s: %s", candidate_email, e)
        return False


# ── Gmail API provider ────────────────────────────────────────────────────────

def _send_via_gmail_api(*, candidate_email: str, subject: str, body: str) -> bool:
    try:
        from Communication_Scheduling.communication_scheduling import get_services
        import base64

        _, gmail_service = get_services()
        mime = EmailMessage()
        mime.set_content(body)
        mime["To"] = candidate_email
        mime["Subject"] = subject
        mime["From"] = os.getenv("RECRUITER_FROM_EMAIL", "me")
        raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
        gmail_service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return True
    except Exception as e:
        logger.warning("Gmail API send failed for %s: %s", candidate_email, e)
        return False


# ── Dispatcher ────────────────────────────────────────────────────────────────

def _send_email(
    *,
    candidate_email: str,
    subject: str,
    body_text: str,
    body_html: Optional[str] = None,
) -> bool:
    if os.getenv("COMMUNICATION_REAL", "").lower() not in ("1", "true", "yes"):
        logger.info("[comm] Dry run — email to %s | %s", candidate_email, subject)
        return True

    provider = os.getenv("COMMUNICATION_PROVIDER", "gmail").strip().lower()

    if provider == "ses":
        html = body_html or _plain_to_html(body_text)
        return _send_via_ses(
            candidate_email=candidate_email,
            subject=subject,
            body_text=body_text,
            body_html=html,
        )
    if provider == "smtp":
        return _send_via_smtp(
            candidate_email=candidate_email,
            subject=subject,
            body=body_text,
        )
    if provider == "gmail":
        return _send_via_gmail_api(
            candidate_email=candidate_email,
            subject=subject,
            body=body_text,
        )

    logger.warning("Unknown COMMUNICATION_PROVIDER=%s; falling back to Gmail API", provider)
    return _send_via_gmail_api(
        candidate_email=candidate_email,
        subject=subject,
        body=body_text,
    )


def _plain_to_html(text: str) -> str:
    """Minimal fallback: wrap plain text in a basic HTML shell."""
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    lines = escaped.split("\n")
    paragraphs = "".join(f"<p>{l}</p>" if l.strip() else "<br>" for l in lines)
    return f"<html><body style='font-family:sans-serif'>{paragraphs}</body></html>"


# ── Email builders ────────────────────────────────────────────────────────────

def send_interview_link_email(
    *,
    candidate_name: str,
    candidate_email: str,
    interview_url: str,
    expires_at: datetime,
    reminder_note: bool = False,
) -> bool:
    subj = "Reminder: Your Interview Link — SmartHire" if reminder_note else "Your Interview Link — SmartHire"

    expiry_str = expires_at.strftime("%B %d, %Y at %I:%M %p UTC")

    body_text = (
        f"Hi {candidate_name},\n\n"
        f"{'This is a reminder that your' if reminder_note else 'Your'} interview link is ready.\n\n"
        f"Interview Link: {interview_url}\n\n"
        f"This link is one-time use and expires on {expiry_str}.\n\n"
        "If you have any issues, please contact our recruitment team.\n\n"
        "Best regards,\nSmartHire Recruitment"
    )

    body_html = f"""
    <html>
    <body style="margin:0;padding:0;background:#f4f4f4;font-family:'Segoe UI',Arial,sans-serif">
      <table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f4;padding:30px 0">
        <tr><td align="center">
          <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08)">
            <!-- Header -->
            <tr>
              <td style="background:#1a1a2e;padding:28px 40px">
                <h1 style="margin:0;color:#ffffff;font-size:22px;font-weight:700;letter-spacing:0.5px">SmartHire</h1>
                <p style="margin:4px 0 0;color:#a0aec0;font-size:13px">Recruitment Platform</p>
              </td>
            </tr>
            <!-- Body -->
            <tr>
              <td style="padding:40px 40px 32px">
                <p style="margin:0 0 16px;font-size:16px;color:#2d3748">Hi <strong>{candidate_name}</strong>,</p>
                <p style="margin:0 0 24px;font-size:15px;color:#4a5568;line-height:1.6">
                  {"This is a reminder that your" if reminder_note else "Your"} self-serve interview link is ready.
                  Please complete your interview at your earliest convenience.
                </p>
                <!-- CTA Button -->
                <table cellpadding="0" cellspacing="0" style="margin:0 0 32px">
                  <tr>
                    <td style="background:#4f46e5;border-radius:6px">
                      <a href="{interview_url}" style="display:inline-block;padding:14px 32px;color:#ffffff;font-size:15px;font-weight:600;text-decoration:none;letter-spacing:0.3px">
                        Start Interview &rarr;
                      </a>
                    </td>
                  </tr>
                </table>
                <!-- Info Box -->
                <table width="100%" cellpadding="0" cellspacing="0" style="background:#f7f8fc;border-radius:6px;margin-bottom:24px">
                  <tr>
                    <td style="padding:16px 20px">
                      <p style="margin:0 0 6px;font-size:13px;color:#718096;text-transform:uppercase;letter-spacing:0.5px;font-weight:600">Important</p>
                      <p style="margin:0;font-size:14px;color:#4a5568">This link is <strong>one-time use</strong> and expires on <strong>{expiry_str}</strong>.</p>
                    </td>
                  </tr>
                </table>
                <p style="margin:0;font-size:13px;color:#a0aec0">
                  If the button above doesn't work, copy and paste this URL into your browser:<br>
                  <a href="{interview_url}" style="color:#4f46e5;word-break:break-all">{interview_url}</a>
                </p>
              </td>
            </tr>
            <!-- Footer -->
            <tr>
              <td style="background:#f7f8fc;padding:20px 40px;border-top:1px solid #e2e8f0">
                <p style="margin:0;font-size:12px;color:#a0aec0;text-align:center">
                  SmartHire Recruitment Platform &mdash; This is an automated message, please do not reply.
                </p>
              </td>
            </tr>
          </table>
        </td></tr>
      </table>
    </body>
    </html>
    """

    sent = _send_email(
        candidate_email=candidate_email,
        subject=subj,
        body_text=body_text,
        body_html=body_html,
    )
    if not sent:
        logger.warning("Interview link email failed; link remains valid for %s", candidate_email)
    return sent


def send_offer_email(
    *,
    candidate_name: str,
    candidate_email: str,
    job_title: str,
) -> bool:
    subj = f"Congratulations — Offer for {job_title}"

    body_text = (
        f"Hi {candidate_name},\n\n"
        f"Congratulations! We are delighted to inform you that you have been selected for the {job_title} role.\n\n"
        "Your offer details are available in the hiring portal. Our team will be in touch with the next steps shortly.\n\n"
        "Best regards,\nSmartHire Recruitment"
    )

    body_html = f"""
    <html>
    <body style="margin:0;padding:0;background:#f4f4f4;font-family:'Segoe UI',Arial,sans-serif">
      <table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f4;padding:30px 0">
        <tr><td align="center">
          <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08)">
            <tr>
              <td style="background:#1a1a2e;padding:28px 40px">
                <h1 style="margin:0;color:#ffffff;font-size:22px;font-weight:700">SmartHire</h1>
                <p style="margin:4px 0 0;color:#a0aec0;font-size:13px">Recruitment Platform</p>
              </td>
            </tr>
            <tr>
              <td style="padding:40px">
                <p style="margin:0 0 8px;font-size:28px">🎉</p>
                <h2 style="margin:0 0 16px;font-size:22px;color:#1a1a2e">Congratulations, {candidate_name}!</h2>
                <p style="margin:0 0 20px;font-size:15px;color:#4a5568;line-height:1.6">
                  We are thrilled to offer you the position of <strong>{job_title}</strong>.
                  You stood out among many strong candidates, and we look forward to having you on board.
                </p>
                <p style="margin:0 0 20px;font-size:15px;color:#4a5568;line-height:1.6">
                  Your offer details are available in the hiring portal. A member of our team will reach out
                  shortly to walk you through the next steps.
                </p>
              </td>
            </tr>
            <tr>
              <td style="background:#f7f8fc;padding:20px 40px;border-top:1px solid #e2e8f0">
                <p style="margin:0;font-size:12px;color:#a0aec0;text-align:center">
                  SmartHire Recruitment Platform &mdash; This is an automated message, please do not reply.
                </p>
              </td>
            </tr>
          </table>
        </td></tr>
      </table>
    </body>
    </html>
    """

    return _send_email(
        candidate_email=candidate_email,
        subject=subj,
        body_text=body_text,
        body_html=body_html,
    )


def send_rejection_email(
    *,
    candidate_name: str,
    candidate_email: str,
    job_title: str,
) -> bool:
    subj = f"Application Update — {job_title}"

    body_text = (
        f"Hi {candidate_name},\n\n"
        f"Thank you for your time and interest in the {job_title} role.\n\n"
        "After careful evaluation, we have decided to move forward with another candidate at this time. "
        "We truly appreciate your effort and encourage you to apply for future opportunities with us.\n\n"
        "Best regards,\nSmartHire Recruitment"
    )

    body_html = f"""
    <html>
    <body style="margin:0;padding:0;background:#f4f4f4;font-family:'Segoe UI',Arial,sans-serif">
      <table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f4;padding:30px 0">
        <tr><td align="center">
          <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08)">
            <tr>
              <td style="background:#1a1a2e;padding:28px 40px">
                <h1 style="margin:0;color:#ffffff;font-size:22px;font-weight:700">SmartHire</h1>
                <p style="margin:4px 0 0;color:#a0aec0;font-size:13px">Recruitment Platform</p>
              </td>
            </tr>
            <tr>
              <td style="padding:40px">
                <h2 style="margin:0 0 16px;font-size:20px;color:#1a1a2e">Application Update</h2>
                <p style="margin:0 0 16px;font-size:15px;color:#4a5568;line-height:1.6">Hi <strong>{candidate_name}</strong>,</p>
                <p style="margin:0 0 16px;font-size:15px;color:#4a5568;line-height:1.6">
                  Thank you for your time and interest in the <strong>{job_title}</strong> role.
                  We genuinely appreciated the opportunity to learn more about your background.
                </p>
                <p style="margin:0;font-size:15px;color:#4a5568;line-height:1.6">
                  After careful consideration, we have decided to move forward with another candidate at this time.
                  We encourage you to apply for future openings that match your skills.
                </p>
              </td>
            </tr>
            <tr>
              <td style="background:#f7f8fc;padding:20px 40px;border-top:1px solid #e2e8f0">
                <p style="margin:0;font-size:12px;color:#a0aec0;text-align:center">
                  SmartHire Recruitment Platform &mdash; This is an automated message, please do not reply.
                </p>
              </td>
            </tr>
          </table>
        </td></tr>
      </table>
    </body>
    </html>
    """

    return _send_email(
        candidate_email=candidate_email,
        subject=subj,
        body_text=body_text,
        body_html=body_html,
    )


def send_prescreen_mcq_email(
    *,
    candidate_name: str,
    candidate_email: str,
    mcq_url: str,
) -> bool:
    subj = "Complete Your Prescreening — SmartHire"

    body_text = (
        f"Hi {candidate_name},\n\n"
        "Before the interview round, please complete the prescreening MCQ using the link below:\n\n"
        f"{mcq_url}\n\n"
        "Once submitted and approved, your interview link will be shared automatically.\n\n"
        "Best regards,\nSmartHire Recruitment"
    )

    body_html = f"""
    <html>
    <body style="margin:0;padding:0;background:#f4f4f4;font-family:'Segoe UI',Arial,sans-serif">
      <table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f4;padding:30px 0">
        <tr><td align="center">
          <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08)">
            <tr>
              <td style="background:#1a1a2e;padding:28px 40px">
                <h1 style="margin:0;color:#ffffff;font-size:22px;font-weight:700">SmartHire</h1>
                <p style="margin:4px 0 0;color:#a0aec0;font-size:13px">Recruitment Platform</p>
              </td>
            </tr>
            <tr>
              <td style="padding:40px">
                <h2 style="margin:0 0 16px;font-size:20px;color:#1a1a2e">Prescreening Step</h2>
                <p style="margin:0 0 16px;font-size:15px;color:#4a5568">Hi <strong>{candidate_name}</strong>,</p>
                <p style="margin:0 0 24px;font-size:15px;color:#4a5568;line-height:1.6">
                  You've been shortlisted! Before proceeding to the interview, please complete the
                  prescreening MCQ. Once submitted and approved, your interview link will be sent automatically.
                </p>
                <table cellpadding="0" cellspacing="0" style="margin:0 0 24px">
                  <tr>
                    <td style="background:#4f46e5;border-radius:6px">
                      <a href="{mcq_url}" style="display:inline-block;padding:14px 32px;color:#ffffff;font-size:15px;font-weight:600;text-decoration:none">
                        Start Prescreening &rarr;
                      </a>
                    </td>
                  </tr>
                </table>
                <p style="margin:0;font-size:13px;color:#a0aec0">
                  Link: <a href="{mcq_url}" style="color:#4f46e5;word-break:break-all">{mcq_url}</a>
                </p>
              </td>
            </tr>
            <tr>
              <td style="background:#f7f8fc;padding:20px 40px;border-top:1px solid #e2e8f0">
                <p style="margin:0;font-size:12px;color:#a0aec0;text-align:center">
                  SmartHire Recruitment Platform &mdash; This is an automated message, please do not reply.
                </p>
              </td>
            </tr>
          </table>
        </td></tr>
      </table>
    </body>
    </html>
    """

    return _send_email(
        candidate_email=candidate_email,
        subject=subj,
        body_text=body_text,
        body_html=body_html,
    )


def send_interview_complete_email(
    *,
    candidate_name: str,
    candidate_email: str,
    job_title: str,
) -> bool:
    subj = f"Interview Received — {job_title}"

    body_text = (
        f"Hi {candidate_name},\n\n"
        f"Thank you for completing your interview for the {job_title} role.\n\n"
        "Our team will carefully review your responses and be in touch with the next steps.\n\n"
        "Best regards,\nSmartHire Recruitment"
    )

    body_html = f"""
    <html>
    <body style="margin:0;padding:0;background:#f4f4f4;font-family:'Segoe UI',Arial,sans-serif">
      <table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f4;padding:30px 0">
        <tr><td align="center">
          <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08)">
            <tr>
              <td style="background:#1a1a2e;padding:28px 40px">
                <h1 style="margin:0;color:#ffffff;font-size:22px;font-weight:700">SmartHire</h1>
                <p style="margin:4px 0 0;color:#a0aec0;font-size:13px">Recruitment Platform</p>
              </td>
            </tr>
            <tr>
              <td style="padding:40px">
                <p style="margin:0 0 8px;font-size:28px">✅</p>
                <h2 style="margin:0 0 16px;font-size:20px;color:#1a1a2e">Interview Submitted</h2>
                <p style="margin:0 0 16px;font-size:15px;color:#4a5568">Hi <strong>{candidate_name}</strong>,</p>
                <p style="margin:0 0 16px;font-size:15px;color:#4a5568;line-height:1.6">
                  Your interview for the <strong>{job_title}</strong> role has been successfully submitted.
                  Our team will carefully review your responses.
                </p>
                <p style="margin:0;font-size:15px;color:#4a5568;line-height:1.6">
                  We will be in touch with the next steps shortly. Thank you for your time!
                </p>
              </td>
            </tr>
            <tr>
              <td style="background:#f7f8fc;padding:20px 40px;border-top:1px solid #e2e8f0">
                <p style="margin:0;font-size:12px;color:#a0aec0;text-align:center">
                  SmartHire Recruitment Platform &mdash; This is an automated message, please do not reply.
                </p>
              </td>
            </tr>
          </table>
        </td></tr>
      </table>
    </body>
    </html>
    """

    return _send_email(
        candidate_email=candidate_email,
        subject=subj,
        body_text=body_text,
        body_html=body_html,
    )


def default_interview_expiry_hours() -> int:
    return int(os.getenv("INTERVIEW_LINK_TTL_HOURS", "72"))


def compute_expiry() -> datetime:
    h = default_interview_expiry_hours()
    return datetime.now(timezone.utc) + timedelta(hours=h)
