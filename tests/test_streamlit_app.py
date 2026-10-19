"""Offline smoke test for the default synthetic dashboard."""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class StreamlitSmokeTests(unittest.TestCase):
    def test_default_demo_renders_without_exceptions_or_network(self) -> None:
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        self.assertEqual(list(app.exception), [])
        self.assertTrue(any(item.value == "Multi-Asset Risk Parity Fund Simulator" for item in app.title))
        self.assertTrue(
            any("SYNTHETIC DEMONSTRATION DATA" in item.value for item in app.warning)
        )
        self.assertTrue(any(item.value == "Backtest" for item in app.subheader))


if __name__ == "__main__":
    unittest.main()
