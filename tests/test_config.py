"""
test_config.py
--------------
Unit tests for configuration and settings validation.
"""

import unittest
from config import Settings


class TestConfig(unittest.TestCase):
    def test_settings_defaults(self):
        s = Settings()
        self.assertEqual(s.app_name, "JANE Research Agent")
        self.assertIn("gemini-2.5-flash-lite", s.available_models)
        self.assertEqual(s.books_dir, "books")
        self.assertEqual(s.scans_dir, "scans")
        self.assertEqual(s.past_chats_dir, "past_chats")
        self.assertGreater(s.http_timeout_s, 0)
        self.assertTrue(s.user_agent.startswith("JANEResearchAgent"))

    def test_directory_ensurer(self):
        s = Settings()
        s.ensure_directories()
        import os
        for folder in [s.books_dir, s.scans_dir, s.past_chats_dir]:
            self.assertTrue(os.path.exists(folder))


if __name__ == "__main__":
    unittest.main()
