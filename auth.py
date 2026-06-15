"""
輕量認證模組（零額外相依，只用標準庫 + requests）

- 使用者資料：users.json（密碼用 pbkdf2-sha256 雜湊）
- Session：HMAC 簽章的 cookie（無狀態，重啟不會登出）
- Gmail 登入：手動串 Google OAuth 2.0（authorization code flow）
"""

import os
import json
import time
import hmac
import base64
import hashlib
import secrets
from pathlib import Path
from urllib.parse import urlencode

import requests

BASE_DIR = Path(__file__).parent
USERS_PATH = BASE_DIR / "users.json"
SECRET_PATH = BASE_DIR / ".session_secret"

# ── Session secret ────────────────────────────────────────────
def _load_secret() -> bytes:
    env = os.environ.get("SESSION_SECRET")
    if env:
        return env.encode("utf-8")
    # 沒設環境變數就自動產生並存檔，重啟後沿用（避免每次重啟把人登出）
    if SECRET_PATH.exists():
        return SECRET_PATH.read_text().strip().encode("utf-8")
    s = secrets.token_hex(32)
    SECRET_PATH.write_text(s)
    return s.encode("utf-8")

SECRET = _load_secret()
SESSION_COOKIE = "session"
SESSION_MAX_AGE = 60 * 60 * 24 * 7  # 7 天

# ── Google OAuth 設定 ─────────────────────────────────────────
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
OAUTH_REDIRECT_BASE = os.environ.get("OAUTH_REDIRECT_BASE", "http://localhost:8867").rstrip("/")
GOOGLE_REDIRECT_URI = OAUTH_REDIRECT_BASE + "/auth/google/callback"

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


def google_enabled() -> bool:
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)


# ── 使用者資料存取 ────────────────────────────────────────────
def _load_users() -> dict:
    if USERS_PATH.exists():
        try:
            return json.loads(USERS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_users(users: dict) -> None:
    USERS_PATH.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")


def _norm_email(email: str) -> str:
    return (email or "").strip().lower()


# ── 使用者識別碼 / 頭貼 ────────────────────────────────────────
def user_uid(email: str) -> str:
    """email 的穩定雜湊（含 SECRET 加鹽），用於對話紀錄關聯，避免明文 PII 落地。"""
    e = _norm_email(email)
    return hmac.new(SECRET, e.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def default_avatar_seed(email: str) -> str:
    """未自訂時，頭貼以正規化 email 當 DiceBear seed。"""
    return _norm_email(email)


def set_avatar_seed(email: str, seed: str) -> str:
    """設定/重骰頭貼 seed；空字串則重置為依 email 自動生成。回傳最終 seed。"""
    users = _load_users()
    e = _norm_email(email)
    if e not in users:
        raise ValueError("使用者不存在")
    final = (seed or "").strip() or default_avatar_seed(e)
    users[e]["avatar_seed"] = final
    _save_users(users)
    return final


def update_name(email: str, name: str) -> dict:
    """更新顯示名稱。"""
    users = _load_users()
    e = _norm_email(email)
    if e not in users:
        raise ValueError("使用者不存在")
    name = (name or "").strip()
    if not name:
        raise ValueError("顯示名稱不可為空")
    users[e]["name"] = name
    _save_users(users)
    return users[e]


# ── 密碼雜湊 ──────────────────────────────────────────────────
def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000).hex()


def get_user(email: str) -> dict | None:
    return _load_users().get(_norm_email(email))


def register_user(email: str, password: str, name: str = "", avatar_seed: str = "") -> dict:
    """建立帳密使用者；email 已存在則拋出 ValueError。"""
    email = _norm_email(email)
    if not email or "@" not in email:
        raise ValueError("請輸入有效的 Email")
    if len(password) < 6:
        raise ValueError("密碼至少需要 6 個字元")
    users = _load_users()
    if email in users:
        raise ValueError("此 Email 已註冊，請直接登入")
    salt = os.urandom(16)
    users[email] = {
        "email": email,
        "name": name or email.split("@")[0],
        "provider": "password",
        "pw_salt": salt.hex(),
        "pw_hash": _hash_password(password, salt),
        "avatar_seed": (avatar_seed or "").strip() or default_avatar_seed(email),
        "email_verified": False,  # 需點驗證信連結才會變 True
        "created": int(time.time()),
    }
    _save_users(users)
    return users[email]


def verify_password(email: str, password: str) -> dict | None:
    user = get_user(email)
    if not user or not user.get("pw_hash"):
        return None
    salt = bytes.fromhex(user["pw_salt"])
    if hmac.compare_digest(_hash_password(password, salt), user["pw_hash"]):
        return user
    return None


def record_consent(email: str) -> None:
    """記錄使用者同意使用條款的時間。"""
    users = _load_users()
    e = _norm_email(email)
    if e in users:
        users[e]["tos_agreed_at"] = int(time.time())
        _save_users(users)


def upsert_google_user(email: str, name: str = "") -> dict:
    """Google 登入：不存在就建立，存在就回傳。"""
    email = _norm_email(email)
    users = _load_users()
    if email not in users:
        users[email] = {
            "email": email,
            "name": name or email.split("@")[0],
            "provider": "google",
            "avatar_seed": default_avatar_seed(email),
            "email_verified": True,  # Google 已驗證過該信箱
            "created": int(time.time()),
        }
        _save_users(users)
    return users[email]


def is_verified(user: dict | None) -> bool:
    return bool(user and user.get("email_verified"))


def mark_verified(email: str) -> bool:
    users = _load_users()
    e = _norm_email(email)
    if e in users:
        users[e]["email_verified"] = True
        _save_users(users)
        return True
    return False


# ── Session cookie 簽章 ───────────────────────────────────────
def _sign(payload: str) -> str:
    raw = payload.encode("utf-8")
    b = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    sig = hmac.new(SECRET, b.encode("ascii"), hashlib.sha256).digest()
    sig_b = base64.urlsafe_b64encode(sig).decode("ascii").rstrip("=")
    return f"{b}.{sig_b}"


def _unsign(token: str) -> str | None:
    try:
        b, sig_b = token.split(".", 1)
    except ValueError:
        return None
    expected = hmac.new(SECRET, b.encode("ascii"), hashlib.sha256).digest()
    expected_b = base64.urlsafe_b64encode(expected).decode("ascii").rstrip("=")
    if not hmac.compare_digest(sig_b, expected_b):
        return None
    pad = "=" * (-len(b) % 4)
    try:
        return base64.urlsafe_b64decode(b + pad).decode("utf-8")
    except Exception:
        return None


def make_session_token(email: str) -> str:
    payload = json.dumps({"email": _norm_email(email), "iat": int(time.time())})
    return _sign(payload)


def read_session_token(token: str | None) -> dict | None:
    if not token:
        return None
    payload = _unsign(token)
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if int(time.time()) - data.get("iat", 0) > SESSION_MAX_AGE:
        return None
    return get_user(data.get("email", ""))


def current_user(request) -> dict | None:
    """從 request 的 cookie 取得目前登入的使用者（或 None）。"""
    return read_session_token(request.cookies.get(SESSION_COOKIE))


# ── Email 驗證 token（簽章、24 小時有效）─────────────────────────
VERIFY_MAX_AGE = 60 * 60 * 24  # 24 小時


def make_verify_token(email: str) -> str:
    payload = json.dumps({"email": _norm_email(email), "purpose": "verify", "iat": int(time.time())})
    return _sign(payload)


def read_verify_token(token: str | None) -> str | None:
    """驗證 token 正確且未過期時，回傳對應 email；否則 None。"""
    if not token:
        return None
    payload = _unsign(token)
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if data.get("purpose") != "verify":
        return None
    if int(time.time()) - data.get("iat", 0) > VERIFY_MAX_AGE:
        return None
    return data.get("email")


# ── Google OAuth helpers ──────────────────────────────────────
def google_auth_url(state: str) -> str:
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    return GOOGLE_AUTH_URL + "?" + urlencode(params)


def google_exchange_code(code: str) -> dict:
    """用 authorization code 換 token，再取得使用者 email/name。"""
    resp = requests.post(GOOGLE_TOKEN_URL, data={
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }, timeout=15)
    resp.raise_for_status()
    access_token = resp.json().get("access_token")
    if not access_token:
        raise RuntimeError("Google 未回傳 access_token")
    info = requests.get(GOOGLE_USERINFO_URL,
                        headers={"Authorization": f"Bearer {access_token}"},
                        timeout=15)
    info.raise_for_status()
    data = info.json()
    email = data.get("email")
    if not email:
        raise RuntimeError("Google 帳號未提供 email")
    return {"email": email, "name": data.get("name", "")}
