# -*- coding: utf-8 -*-
"""Quantitative accuracy tests for StockTrendAnalyzer."""

import unittest

import pandas as pd

from src.stock_analyzer import StockTrendAnalyzer, TrendAnalysisResult, TrendStatus


class StockAnalyzerQuantitativeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = StockTrendAnalyzer()

    def test_atr_is_calculated_from_true_range(self) -> None:
        df = pd.DataFrame(
            {
                "close": [10.0, 12.0, 11.0],
                "high": [11.0, 13.0, 12.0],
                "low": [9.0, 11.0, 10.0],
            }
        )

        result = self.analyzer._calculate_atr(df)

        self.assertAlmostEqual(float(result["ATR14"].iloc[-1]), 2.3333333333333335)

    def test_support_tolerance_expands_with_atr_but_is_capped(self) -> None:
        low_vol = self.analyzer._adaptive_support_tolerance(
            type("Result", (), {"atr_pct": 1.0})()
        )
        high_vol = self.analyzer._adaptive_support_tolerance(
            type("Result", (), {"atr_pct": 8.0})()
        )

        self.assertAlmostEqual(low_vol, self.analyzer.MA_SUPPORT_TOLERANCE)
        self.assertAlmostEqual(high_vol, 0.05)

    def test_ma_support_uses_intraday_touch_and_close_recovery(self) -> None:
        self.assertTrue(self.analyzer._has_ma_support(99.0, 96.5, 100.0, 0.03))
        self.assertFalse(self.analyzer._has_ma_support(94.0, 96.5, 100.0, 0.03))

    def test_dynamic_trend_strength_distinguishes_uptrend_from_flat_market(self) -> None:
        dates = pd.date_range("2026-01-01", periods=60, freq="D")
        up_close = [100.0 + i * 0.8 for i in range(60)]
        flat_close = [100.0 + (0.2 if i % 2 else -0.2) for i in range(60)]

        up = self.analyzer.analyze(self._ohlcv(dates, up_close), "UP")
        flat = self.analyzer.analyze(self._ohlcv(dates, flat_close), "FLAT")

        self.assertGreater(up.trend_strength, flat.trend_strength)
        self.assertGreater(up.ma20_slope_pct, 0)
        self.assertGreater(up.price_return_20d, 0)

    def test_trend_score_uses_strength_with_status_bounds(self) -> None:
        weak_bull = TrendAnalysisResult(
            code="TEST",
            trend_status=TrendStatus.BULL,
            trend_strength=62,
        )
        strong_bull = TrendAnalysisResult(
            code="TEST",
            trend_status=TrendStatus.BULL,
            trend_strength=92,
        )

        self.assertLess(
            self.analyzer._trend_score_from_strength(weak_bull),
            self.analyzer._trend_score_from_strength(strong_bull),
        )
        self.assertLessEqual(self.analyzer._trend_score_from_strength(strong_bull), 28)

    @staticmethod
    def _ohlcv(dates: pd.DatetimeIndex, closes: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": dates,
                "open": [price * 0.995 for price in closes],
                "high": [price * 1.01 for price in closes],
                "low": [price * 0.99 for price in closes],
                "close": closes,
                "volume": [1_000_000 + i * 1000 for i in range(len(closes))],
            }
        )


if __name__ == "__main__":
    unittest.main()
