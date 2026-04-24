import unittest

from flow import classify_ks


class FlowRuleTests(unittest.TestCase):
    def test_classify_ks_below(self) -> None:
        self.assertEqual(classify_ks(80.0), "KS 110±5 미달")

    def test_classify_ks_in_range(self) -> None:
        self.assertEqual(classify_ks(110.0), "KS 범위")

    def test_classify_ks_above(self) -> None:
        self.assertEqual(classify_ks(123.2), "KS 초과")


if __name__ == "__main__":
    unittest.main()
