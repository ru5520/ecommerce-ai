"""商家后台密码校验。"""

import os
import hmac


def password_configured() -> bool:
    return bool(os.environ.get("ADMIN_PASSWORD", "").strip())


def verify_password(input_password: str) -> bool:
    expected = os.environ.get("ADMIN_PASSWORD", "").strip()
    if not expected:
        return False
    return hmac.compare_digest(str(input_password or ""), expected)
