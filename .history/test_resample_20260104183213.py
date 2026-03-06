"""Debug: Check if the engine is correctly using different frequencies"""

import pandas as pd
import yfinance as yf


def main() -> None:
	print("Downloading GLD...")
	data = yf.download("GLD", start="2005-01-01", end="2025-12-31", progress=False)
	prices = data["Close"]

	print(f"\nDaily prices: {len(prices)} observations")
	print(f"Date range: {prices.index[0]} to {prices.index[-1]}")

	# Resample to different frequencies
	monthly = prices.resample("ME").last()
	weekly = prices.resample("W").last()
	biweekly = prices.resample("2W").last()

	print(f"\nMonthly (ME): {len(monthly)} periods")
	print(f"Weekly (W): {len(weekly)} periods")
	print(f"Biweekly (2W): {len(biweekly)} periods")

	# Calculate buy-and-hold returns
	monthly_ret = monthly.pct_change().dropna()
	weekly_ret = weekly.pct_change().dropna()
	biweekly_ret = biweekly.pct_change().dropna()

	cum_monthly = float((1 + monthly_ret).cumprod().iloc[-1])
	cum_weekly = float((1 + weekly_ret).cumprod().iloc[-1])
	cum_biweekly = float((1 + biweekly_ret).cumprod().iloc[-1])

	print("\nFinal cumulative (buy & hold):")
	print(f"  Monthly: {cum_monthly:.4f}")
	print(f"  Weekly: {cum_weekly:.4f}")
	print(f"  Biweekly: {cum_biweekly:.4f}")

	print("\nThese should be DIFFERENT if resampling works correctly!")
	print(f"Weekly vs Monthly ratio: {cum_weekly / cum_monthly:.6f}")
	print(f"Biweekly vs Monthly ratio: {cum_biweekly / cum_monthly:.6f}")


if __name__ == "__main__":
	main()
