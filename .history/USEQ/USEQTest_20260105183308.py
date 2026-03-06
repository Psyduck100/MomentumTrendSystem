    # ----------------------------
    # Return contribution by held ticker (log-return attribution)
    # ----------------------------
    rets = res["out"]["port_ret"].dropna()
    hold_applied = res["out"]["holding"].reindex(rets.index).astype(str)

    # avg monthly return while held + count months
    by_ticker = rets.groupby(hold_applied).agg(months="count", avg_monthly_ret="mean")

    # Contribution to total compounded return using log(1+r)
    # (log-returns add up cleanly across time and partition by ticker)
    logret = np.log1p(rets)
    sum_logret = logret.groupby(hold_applied).sum().rename("sum_logret")

    by_ticker = by_ticker.join(sum_logret)

    total_log = by_ticker["sum_logret"].sum()
    by_ticker["pct_total_logret"] = by_ticker["sum_logret"] / total_log if total_log != 0 else np.nan

    # Pretty formatting
    by_ticker["avg_monthly_ret"] = by_ticker["avg_monthly_ret"].map(lambda x: f"{x:.2%}")
    by_ticker["pct_total_logret"] = by_ticker["pct_total_logret"].map(lambda x: f"{x:.1%}")

    print("\n" + "=" * 70)
    print("Return contribution by ticker (based on months held)")
    print("=" * 70)
    print(by_ticker.sort_values("sum_logret", ascending=False).to_string())

    # ----------------------------
    # Average holding streak length (how long you keep a ticker once selected)
    # ----------------------------
    h = res["out"]["holding"].dropna().astype(str)

    # Identify streak boundaries
    streak_id = (h != h.shift(1)).cumsum()
    streaks = pd.DataFrame({"ticker": h, "streak_id": streak_id})
    streak_lengths = streaks.groupby(["ticker", "streak_id"]).size().rename("streak_len").reset_index()

    streak_stats = streak_lengths.groupby("ticker")["streak_len"].agg(
        streaks="count",
        avg_streak="mean",
        median_streak="median",
        max_streak="max",
    ).sort_values("avg_streak", ascending=False)

    # Pretty formatting
    streak_stats["avg_streak"] = streak_stats["avg_streak"].map(lambda x: f"{x:.2f}")
    streak_stats["median_streak"] = streak_stats["median_streak"].map(lambda x: f"{x:.0f}")
    streak_stats["max_streak"] = streak_stats["max_streak"].map(lambda x: f"{x:.0f}")

    print("\n" + "=" * 70)
    print("Holding streak stats (months per continuous hold)")
    print("=" * 70)
    print(streak_stats.to_string())
