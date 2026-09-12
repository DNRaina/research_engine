"""
test_tools.py
-------------
Unit tests for data sanitization and tool response parsing.
"""

import unittest
from mcp_tools import _strip_html


class TestTools(unittest.TestCase):
    def test_strip_html(self):
        html_input = '<span class="searchmatch">Artificial</span> <b>Intelligence</b> & Machine Learning'
        stripped = _strip_html(html_input)
        self.assertEqual(stripped, "Artificial Intelligence & Machine Learning")

    def test_strip_empty(self):
        self.assertEqual(_strip_html(""), "")
        self.assertEqual(_strip_html(None), "")


if __name__ == "__main__":
    unittest.main()
