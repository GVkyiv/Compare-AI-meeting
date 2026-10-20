import unittest

from compare import SECTION_TITLES, build_user_message, calc_cost, check_protocol, estimate_tokens


class CompareHelpersTest(unittest.TestCase):
    def test_estimate_tokens_is_deterministic(self):
        self.assertEqual(estimate_tokens("a" * 260), 100)

    def test_calc_cost_uses_prompt_and_completion_prices(self):
        prices = {"model": (0.001, 0.002)}
        self.assertEqual(calc_cost("model", 1000, 500, prices), 2.0)

    def test_check_protocol_finds_all_sections_and_metrics(self):
        text = "\n".join(
            f"## {number}. {title}\nУточнити 01.02.2026\n|---|"
            for number, title in SECTION_TITLES.items()
        )
        result = check_protocol(text)
        self.assertEqual(result["sections"], list(SECTION_TITLES))
        self.assertEqual(result["missing"], [])
        self.assertEqual(result["utochnyty"], len(SECTION_TITLES))
        self.assertEqual(result["dates"], 1)

    def test_build_user_message_replaces_template_fields(self):
        result = build_user_message("[Date] [Users] [Transcription]", "text", "01.02.2026", "A")
        self.assertEqual(result, "01.02.2026 A text")


if __name__ == "__main__":
    unittest.main()
