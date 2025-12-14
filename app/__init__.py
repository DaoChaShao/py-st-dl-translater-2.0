#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/5 20:17
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py.py
# @Desc     :

"""
****************************************************************
Next Word Prediction Streamlit Application Package
----------------------------------------------------------------
This package provides:

+ subpages: All Streamlit subpage modules (home, prediction, etc.).
+ tools: Page configuration, layout control, and navigation utilities.

It serves as the entry point for loading app modules and exposing
core subpackages for high-level import and integration.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.2.0"

from . import subpages
from . import tools

__all__ = [
    "subpages",
    "tools",
]
