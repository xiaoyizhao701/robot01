#!/usr/bin/env python3
import json
from pathlib import Path
import unittest

from supermarket_sorting_server import fixed_baseline_layout


TASK_DIR = Path(__file__).resolve().parent


class FixedBaselineLayoutTest(unittest.TestCase):
    def test_target_is_the_only_cola_on_its_shelf_row(self):
        original = json.loads((TASK_DIR / "retail_competition_layout.json").read_text())
        layout, overrides = fixed_baseline_layout(original)

        target = next(slot for slot in layout if slot["body"] == "product_032")
        self.assertEqual((target["shelf"], target["level"], target["column"]),
                         ("D", "L2", "C2"))
        self.assertEqual(target["world_position"], [0.92, 3.243, 0.9235])
        row_kele = [
            slot for slot in layout
            if slot["shelf"] == "D" and slot["level"] == "L2"
            and slot["object_kind"] == "kele"
        ]
        self.assertEqual([slot["body"] for slot in row_kele], ["product_032"])
        self.assertNotIn("product_032", overrides)
        self.assertEqual(len(overrides), 2)

        original_target = next(
            slot for slot in original if slot["body"] == "product_032"
        )
        self.assertEqual(original_target["world_position"], [0.92, 3.243, 0.9235])


if __name__ == "__main__":
    unittest.main()
