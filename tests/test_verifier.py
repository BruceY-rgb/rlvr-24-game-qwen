import unittest

from twentyfour_rl.rewards import score_output
from twentyfour_rl.verifier import extract_answer, has_r1_format, verify


class VerifierTest(unittest.TestCase):
    def test_valid_expression(self):
        result = verify([3, 3, 8, 8], "8/(3-8/3)")
        self.assertTrue(result["is_valid"])
        self.assertTrue(result["is_correct"])
        self.assertEqual(result["error_type"], "ok")

    def test_repeated_number_rejected(self):
        result = verify([1, 2, 3, 4], "1+2+3+3")
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["error_type"], "number_mismatch")

    def test_missing_number_rejected(self):
        result = verify([1, 2, 3, 4], "1+2+3")
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["error_type"], "number_mismatch")

    def test_concatenated_number_rejected(self):
        result = verify([1, 2, 3, 4], "12+3+4")
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["error_type"], "number_mismatch")

    def test_illegal_character_rejected(self):
        result = verify([1, 2, 3, 4], "__import__('os').system('ls')")
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["error_type"], "illegal_character")

    def test_division_by_zero_rejected(self):
        result = verify([1, 1, 2, 2], "2/(1-1)+2")
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["error_type"], "division_by_zero")

    def test_float_literal_rejected(self):
        result = verify([1, 2, 3, 4], "1.0+2+3+4")
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["error_type"], "float_literal")

    def test_extract_answer(self):
        raw = "<think>x</think><answer>(1+2+3)*4</answer>"
        self.assertEqual(extract_answer(raw), "(1+2+3)*4")
        self.assertTrue(has_r1_format(raw))

    def test_reward(self):
        raw = "<think>x</think><answer>8/(3-8/3)</answer>"
        scored = score_output([3, 3, 8, 8], raw)
        self.assertGreaterEqual(scored["reward"], 1.0)
        self.assertEqual(scored["format"], 1.0)


if __name__ == "__main__":
    unittest.main()

