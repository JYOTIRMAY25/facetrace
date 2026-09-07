"""HTTP API package with compatibility exports for the existing prototype."""


def __getattr__(name: str):
    if name in {"app", "verification_engine"}:
        from ..main import app, verification_engine

        return {"app": app, "verification_engine": verification_engine}[name]
    raise AttributeError(name)
