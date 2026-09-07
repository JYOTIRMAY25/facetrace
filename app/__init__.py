"""FaceTrace — face-derived content discovery with blockchain-anchored verification.

Pipeline (see ../../SYSTEM_ARCHITECTURE.md):

    face image -> detection -> encoding -> genuine web/social search
    -> real matching post -> canonicalize -> SHA-256 fingerprint
    -> blockchain write -> blockchain read -> recompute -> compare
    -> VERIFIED / NOT VERIFIED

Scope note (read before using this package):

* Only process images and content you have permission to process.
* A positive result reports that *content matching the submitted image was
  found and that its fingerprint is unchanged since it was anchored*. It does
  NOT establish, prove, or assert anyone's legal identity.

STEP 1 status: interfaces, configuration, and structured result models only.
No face recognition, search, or blockchain behaviour is implemented yet.
"""

__all__ = ["__version__", "CONTENT_MATCH_DISCLAIMER"]

__version__ = "0.1.0"

CONTENT_MATCH_DISCLAIMER = (
    "FaceTrace reports a CONTENT MATCH and a fingerprint integrity check. "
    "It does not establish legal identity. Only process content you are "
    "authorised to process."
)
