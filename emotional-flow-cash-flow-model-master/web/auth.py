# -*- coding: utf-8 -*-
"""认证工具: 密码哈希 + 图片验证码"""
import io
import random
import string
import bcrypt
from captcha.image import ImageCaptcha

_image_captcha = ImageCaptcha(width=160, height=60)

# 字符集: 去除易混淆字符 0/O/1/I
_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def gen_captcha(length: int = 4):
    """生成验证码 (code, PNG 字节流)"""
    code = "".join(random.choices(_CHARS, k=length))
    img = _image_captcha.generate_image(code)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return code, buf
