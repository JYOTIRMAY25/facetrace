"""Security-related constants for the backend foundation."""

ALLOWED_IMAGE_TYPES = {
    "image/jpeg": {b"\xff\xd8\xff"},
    "image/png": {b"\x89PNG\r\n\x1a\n"},
    "image/webp": {b"RIFF"},
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
