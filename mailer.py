"""
輕量寄信模組（只用標準庫 smtplib）。

- 設定來自環境變數（.env）：SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASS / SMTP_FROM
- Gmail 範例：SMTP_HOST=smtp.gmail.com、SMTP_PORT=587、SMTP_USER=你的@gmail.com、
  SMTP_PASS=應用程式密碼（需先開啟兩步驟驗證後產生）
- 未設定完整 SMTP 時，不會真的寄出，改把內容印到主控台（方便本機開發直接點連結）。
"""

import os
import ssl
import smtplib
from email.message import EmailMessage
from email.utils import formataddr


def _cfg() -> dict:
    return {
        "host": os.environ.get("SMTP_HOST", "").strip(),
        "port": int(os.environ.get("SMTP_PORT", "587") or "587"),
        "user": os.environ.get("SMTP_USER", "").strip(),
        "password": os.environ.get("SMTP_PASS", "").strip(),
        "from": (os.environ.get("SMTP_FROM", "").strip() or os.environ.get("SMTP_USER", "").strip()),
        "from_name": os.environ.get("SMTP_FROM_NAME", "TaDELS RAG").strip(),
    }


def mail_enabled() -> bool:
    c = _cfg()
    return bool(c["host"] and c["user"] and c["password"])


def send_email(to: str, subject: str, html: str, text: str = "") -> bool:
    """寄一封信。回傳 True=真的寄出；False=未設定 SMTP（已印到主控台）。

    寄送失敗會丟出例外，由呼叫端決定如何處理。
    """
    c = _cfg()
    if not mail_enabled():
        print("=" * 64)
        print("[MAIL · 尚未設定 SMTP，僅印出內容（請直接複製連結）]")
        print("To     :", to)
        print("Subject:", subject)
        print("-" * 64)
        print(text or html)
        print("=" * 64)
        return False

    msg = EmailMessage()
    msg["From"] = formataddr((c["from_name"], c["from"]))
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(text or "請以支援 HTML 的郵件軟體開啟此信。")
    if html:
        msg.add_alternative(html, subtype="html")

    ctx = ssl.create_default_context()
    if c["port"] == 465:
        with smtplib.SMTP_SSL(c["host"], c["port"], context=ctx, timeout=20) as s:
            s.login(c["user"], c["password"])
            s.send_message(msg)
    else:
        with smtplib.SMTP(c["host"], c["port"], timeout=20) as s:
            s.ehlo()
            s.starttls(context=ctx)
            s.login(c["user"], c["password"])
            s.send_message(msg)
    return True
