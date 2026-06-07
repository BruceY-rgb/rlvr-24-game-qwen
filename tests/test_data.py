import unittest

from twentyfour_rl.data import extract_numbers, extract_target, normalize_row
from twentyfour_rl.prompts import build_user_prompt


class DataParsingTest(unittest.TestCase):
    def test_countdown_row_with_three_numbers(self):
        row = normalize_row(
            {"target": 98, "nums": [44, 19, 35]},
            split="train",
            source="countdown",
            idx=0,
            solvable_default=True,
            min_numbers=3,
            max_numbers=4,
            min_card_value=1,
            max_card_value=100,
            target_from_example=True,
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["numbers"], [44, 19, 35])
        self.assertEqual(row["target"], 98)
        self.assertIn("equals 98", row["prompt"])

    def test_24_row_rejects_three_numbers(self):
        row = normalize_row(
            {"target": 24, "nums": [3, 8, 8]},
            split="train",
            source="bad_24",
            idx=0,
            min_numbers=4,
            max_numbers=4,
            min_card_value=1,
            max_card_value=13,
        )
        self.assertIsNone(row)

    def test_extract_target(self):
        self.assertEqual(extract_target({"target": "target is 57"}), 57)
        self.assertEqual(extract_target({}, default=24), 24)

    def test_prompt_no_longer_says_four(self):
        prompt = build_user_prompt([44, 19, 35], 98)
        self.assertIn("Given these integers", prompt)
        self.assertNotIn("four integers", prompt)

    def test_extract_numbers_from_text(self):
        self.assertEqual(extract_numbers({"problem": "Cards: 44, 19, 35"}, min_numbers=3, max_numbers=4), [44, 19, 35])


if __name__ == "__main__":
    unittest.main()
