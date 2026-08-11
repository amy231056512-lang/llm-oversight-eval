import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import openai

os.environ.setdefault("GROQ_API_KEY", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline import DATASET_PATH, call_with_retry, load_dataset


class DeliberationDatasetTest(unittest.TestCase):
    def test_load_dataset_reads_the_jsonl_dataset(self) -> None:
        cases = load_dataset()

        self.assertGreater(len(cases), 0)
        self.assertGreaterEqual(len(cases), 30)
        self.assertEqual(cases[0]["id"], "dbg_000")

    def test_dataset_path_points_to_the_main_dataset_file(self) -> None:
        self.assertTrue(DATASET_PATH.exists())
        self.assertIn(DATASET_PATH.name, {"dataset.jsonl", "pilot_10cases.jsonl"})

    def test_call_with_retry_uses_api_retry_after_for_rate_limits(self) -> None:
        response = Mock()
        response.headers = {"Retry-After": "7"}
        rate_limit_error = openai.RateLimitError(
            message="Rate limit reached",
            response=response,
            body=None,
        )

        success_completion = Mock()
        success_completion.choices = [Mock(message=Mock(content="{}"))]
        success_completion.model = "openai/gpt-oss-120b"

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [rate_limit_error, success_completion]

        with patch("src.baseline.client", mock_client), patch("src.baseline.time.sleep") as sleep_mock:
            content, actual_model = call_with_retry("openai/gpt-oss-120b", "prompt", max_retries=2)

        self.assertEqual(content, "{}")
        self.assertEqual(actual_model, "openai/gpt-oss-120b")
        self.assertEqual(sleep_mock.call_count, 1)
        sleep_mock.assert_called_once_with(7.0)


if __name__ == "__main__":
    unittest.main()
