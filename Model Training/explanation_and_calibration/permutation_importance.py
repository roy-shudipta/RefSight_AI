#!/usr/bin/env python3
"""Full-series and event-aligned windowed permutation are run here."""

import listwise_explanation_core as core


core.PERM_ENABLE = True
core.WINDOW_ENABLE = True
core.RISK_ENABLE = False


if __name__ == "__main__":
    core.main()
