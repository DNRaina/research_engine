"""
test_resilience.py
------------------
Unit tests for the MCP resilience wrapper (timeouts, retries, non-blocking executor).
"""

import time
import unittest
from mcp_server import _tool_wrapper


class TestResilience(unittest.TestCase):
    def test_successful_call(self):
        def dummy_fn(x, y):
            return f"result: {x + y}"

        wrapped = _tool_wrapper(dummy_fn, timeout_s=1.0, max_attempts=1)
        res = wrapped(3, 4)
        self.assertEqual(res, "result: 7")

    def test_timeout_unblocks_promptly(self):
        def slow_fn():
            time.sleep(2.0)
            return "too late"

        wrapped = _tool_wrapper(slow_fn, timeout_s=0.2, max_attempts=1)
        t0 = time.perf_counter()
        res = wrapped()
        elapsed = time.perf_counter() - t0

        self.assertEqual(res, "")
        # Must return in ~0.2s, not 2.0s (verifies executor doesn't block on shutdown)
        self.assertLess(elapsed, 1.0)

    def test_retry_eventual_success(self):
        calls = {"count": 0}

        def flaky_fn():
            calls["count"] += 1
            if calls["count"] < 2:
                raise ValueError("transient error")
            return "recovered"

        wrapped = _tool_wrapper(flaky_fn, timeout_s=1.0, max_attempts=3, base_delay=0.1)
        res = wrapped()
        self.assertEqual(res, "recovered")
        self.assertEqual(calls["count"], 2)


if __name__ == "__main__":
    unittest.main()
