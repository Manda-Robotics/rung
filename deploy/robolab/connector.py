# SPDX-License-Identifier: Apache-2.0
"""Verified policy entry point. Copy into Verified's policies/rung; the code lives in rung."""


def __getattr__(name):
    if name == "RungPolicyClient":
        from rung.adapters.robolab.client import RungPolicyClient

        return RungPolicyClient
    raise AttributeError(name)
