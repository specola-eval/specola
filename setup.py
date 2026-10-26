"""Build the optional Linux bridge used by guarded native TLS observation."""

import sys

from setuptools import Extension, setup


setup(ext_modules=[Extension(
    "specola._native_tls",
    sources=["src/specola/_native_tls.c"],
    optional=True,
)] if sys.platform == "linux" else [])
