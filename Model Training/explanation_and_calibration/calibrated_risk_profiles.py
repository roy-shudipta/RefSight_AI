#!/usr/bin/env python3
"""Validation calibration and held-out calibrated risk profiles are run here."""

import listwise_explanation_core as core


core.PERM_ENABLE = False
core.WINDOW_ENABLE = False
core.RISK_ENABLE = True


if __name__ == "__main__":
    core.main()
