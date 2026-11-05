#!/usr/bin/env python3
"""Signal scan (trader-scanner): Heikin Ashi + Supertrend + MACD.

A chart matches when ALL three hold on the last COMPLETED bar:

  1. Heikin Ashi: the last two candles are green (HA close > HA open).
     By default the candle before them is not green, so it fires on the
     first two greens of a new run (--any-ha drops that).
  2. Supertrend (ATR 10, factor 3, the TradingView defaults) is bullish:
     the close is above the Supertrend line.
  3. MACD (12, 26, 9) crossed ABOVE its signal line while BELOW zero,
     on one of the last --cross-within bars (3 by default, 1 = this bar only).

Two stages, so the whole market stays quick:
  Stage 1, every symbol: Yahoo spark daily closes, 20 a request. Liquidity
           filter, then the MACD rule (closes are all MACD needs).
  Stage 2, each MACD candidate: the v8 chart with real open, high, low and
           close for Heikin Ashi and Supertrend, one request each.

A match is a chart that matched a rule, not a buy recommendation.
"""
import argparse
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scan  # noqa: E402  (the scanner's Yahoo, calendar and symbol helpers)

NOTE = scan.NOTE
MIN_BARS = 60


# ---------------------------------------------------------------- indicators (pure)

def ema(values, n):
    """EMA seeded with the SMA of the first n values (as TradingView does)."""
    out = [None] * len(values)
    if len(values) < n:
        return out
    k = 2.0 / (n + 1)
    e = sum(values[:n]) / n
    out[n - 1] = e
    for i in range(n, len(values)):
        e = values[i] * k + e * (1 - k)
        out[i] = e
    return out


def macd(closes, fast=12, slow=26, signal=9):
    """-> (macd_line, signal_line), None where not yet defined."""
    ef, es = ema(closes, fast), ema(closes, slow)
    line = [None if a is None or b is None else a - b for a, b in zip(ef, es)]
    start = next((i for i, v in enumerate(line) if v is not None), None)
    sig = [None] * len(closes)
    if start is not None:
        s = ema(line[start:], signal)
        sig[start:] = s
    return line, sig


def macd_cross_below_zero(closes, within=3, fast=12, slow=26, signal=9):
    """Bars ago (0 = last bar) of the latest MACD-over-signal cross made below
    zero within the last `within` bars, else None."""
    line, sig = macd(closes, fast, slow, signal)
    n = len(closes)
    for ago in range(within):
        i = n - 1 - ago
        if i < 1 or None in (line[i], sig[i], line[i - 1], sig[i - 1]):
            break
        if line[i - 1] <= sig[i - 1] and line[i] > sig[i] and line[i] < 0 and sig[i] < 0:
            return ago
    return None


def heikin_ashi(o, h, l, c):
    """-> (ha_open, ha_close)."""
    hc = [(o[i] + h[i] + l[i] + c[i]) / 4.0 for i in range(len(c))]
    ho = [None] * len(c)
    if c:
        ho[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(c)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2.0
    return ho, hc


def atr_rma(h, l, c, n):
    tr = [h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(1, len(c))]
    out = [None] * len(c)
    if len(c) < n:
        return out
    a = sum(tr[:n]) / n
    out[n - 1] = a
    for i in range(n, len(c)):
        a = (a * (n - 1) + tr[i]) / n
        out[i] = a
    return out


def supertrend(h, l, c, n=10, factor=3.0):
    """TradingView's ta.supertrend. -> (line, bullish) lists; bullish True when
    the trend is up (TradingView direction -1)."""
    atr = atr_rma(h, l, c, n)
    line, bull = [None] * len(c), [None] * len(c)
    upper = lower = None
    prev_line = None
    for i in range(len(c)):
        if atr[i] is None:
            continue
        mid = (h[i] + l[i]) / 2.0
        up, dn = mid - factor * atr[i], mid + factor * atr[i]
        if lower is None:  # first bar with an ATR: start in a downtrend, as Pine does
            lower, upper = up, dn
            bull[i], line[i] = False, upper
            prev_line = upper
            continue
        pc = c[i - 1]
        new_lower = up if (up > lower or pc < lower) else lower
        new_upper = dn if (dn < upper or pc > upper) else upper
        if prev_line == upper:
            b = c[i] > new_upper
        else:
            b = not (c[i] < new_lower)
        lower, upper = new_lower, new_upper
        line[i] = lower if b else upper
        bull[i] = b
        prev_line = line[i]
    return line, bull


def check_signal(bars, within=3, fresh=True, st_len=10, st_factor=3.0):
    """bars = [(date, o, h, l, c)]. Pure. -> dict of readings, 'match' True/False."""
    o = [b[1] for b in bars]
    h = [b[2] for b in bars]
    l = [b[3] for b in bars]
    c = [b[4] for b in bars]
    ho, hc = heikin_ashi(o, h, l, c)
    green = [hc[i] > ho[i] for i in range(len(c))]
    two_green = len(c) >= 3 and green[-1] and green[-2] and (not fresh or not green[-3])
    st_line, st_bull = supertrend(h, l, c, st_len, st_factor)
    cross_ago = macd_cross_below_zero(c, within)
    line, sig = macd(c)
    return {
        "match": bool(two_green and st_bull[-1] and cross_ago is not None),
        "ha_two_green": bool(two_green),
        "supertrend_bull": bool(st_bull[-1]),
        "supertrend": round(st_line[-1], 2) if st_line[-1] is not None else None,
        "macd_cross_bars_ago": cross_ago,
        "macd": round(line[-1], 4) if line[-1] is not None else None,
        "macd_signal": round(sig[-1], 4) if sig[-1] is not None else None,
    }


# ---------------------------------------------------------------- data

def complete_ohlc(series, now_utc):
    """[(date, o, h, l, c)] of complete daily bars (today's dropped until 4:00 PM)."""
    now = scan.to_eastern(now_utc)
    today = now.date()
    s = scan.session(today)
    today_done = s is None or now.time() >= s[1]
    out = {}
    for i, ts in enumerate(series["t"]):
        vals = [(series.get(k) or [None] * len(series["t"]))[i] for k in ("o", "h", "l", "c")]
        if any(v is None for v in vals):
            continue
        d = scan.ts_to_eastern(ts).date()
        if d > today or (d == today and not today_done):
            continue
        out[d] = tuple(float(v) for v in vals)
    return [(d,) + out[d] for d in sorted(out)]


def liquid(series, min_price, min_volume, closes):
    meta = series.get("meta") or {}
    price = meta.get("regularMarketPrice")
    if price is None and closes:
        price = closes[-1][1]
    vol = meta.get("regularMarketVolume")
    ok = not (price is None or price < min_price or (min_volume > 0 and (vol is None or vol < min_volume)))
    return ok, price, vol


# ---------------------------------------------------------------- command

def run(a):
    t0 = time.time()
    now = scan.parse_now(a.now)
    symbols = scan.read_symbols(a)
    fresh = not a.any_ha
    scan.progress(a, "Stage 1: daily closes for %s symbols (%d spark requests)..."
                  % ("{:,}".format(len(symbols)), (len(symbols) + scan.SPARK_MAX - 1) // scan.SPARK_MAX))
    data, failed = scan.fetch_spark(symbols, "1y", "1d", a.pause, a)

    counts = {"requested": len(symbols), "fetched": len(data), "liquid": 0, "short": 0,
              "macd_candidates": 0, "confirmed": 0, "stage2_failed": 0}
    cands = []
    for sym in symbols:
        s = data.get(sym)
        if not s:
            continue
        closes = scan.daily_closes(s, now)
        ok, price, vol = liquid(s, a.min_price, a.min_volume, closes)
        if not ok:
            continue
        counts["liquid"] += 1
        if len(closes) < MIN_BARS:
            counts["short"] += 1
            continue
        if macd_cross_below_zero([x[1] for x in closes], a.cross_within) is None:
            continue
        cands.append((sym, price, vol))
    counts["macd_candidates"] = len(cands)
    scan.progress(a, "Stage 2: %d MACD candidates, fetching real OHLC (one v8 chart each)..." % len(cands))

    hits, dates = [], {}
    for k, (sym, price, vol) in enumerate(cands):
        if k and a.pause:
            scan.SLEEP(a.pause)
        try:
            series = scan.parse_chart(scan.yahoo_get(scan.chart_path(sym, "1y", "1d")), sym)
        except scan.ScanError:
            counts["stage2_failed"] += 1
            continue
        bars = complete_ohlc(series, now)
        if len(bars) < MIN_BARS:
            counts["stage2_failed"] += 1
            continue
        counts["confirmed"] += 1
        d = bars[-1][0]
        dates[d] = dates.get(d, 0) + 1
        r = check_signal(bars, a.cross_within, fresh, a.st_length, a.st_factor)
        if not r["match"]:
            continue
        r.pop("match")
        hits.append(dict({"symbol": sym, "price": round(bars[-1][4], 2), "close_date": d.isoformat(),
                          "volume": int(vol) if vol is not None else None}, **r))
    hits.sort(key=lambda h: (-(h["price"] * (h["volume"] or 0)), h["symbol"]))
    last = max(dates, key=lambda d: (dates[d], d)) if dates else None

    rule = ("Heikin Ashi: last 2 candles green%s; Supertrend(%d, %g) bullish; MACD(12,26,9) crossed above "
            "signal below zero within the last %d bar%s" % (
                " (the one before red)" if fresh else "", a.st_length, a.st_factor,
                a.cross_within, "" if a.cross_within == 1 else "s"))
    doc = {"created_by": "trader-scanner/signals", "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "mode": "signal_ha_supertrend_macd", "rule": rule,
           "as_of": ("%s close" % last.isoformat()) if last else None,
           "universe": len(symbols), "scanned": counts["confirmed"], "failed": len(failed),
           "source": "yahoo", "filters": {"min_price": a.min_price, "min_volume": a.min_volume},
           "note": NOTE, "hits": hits}
    if not a.no_write:
        scan.write_json(a.out, doc)
        if a.tv_out:
            d = os.path.dirname(a.tv_out)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(a.tv_out, "w", encoding="utf-8") as fh:
                fh.write(",".join(h["symbol"].replace(".", "_") for h in hits) + "\n")
    if a.json:
        print(json.dumps(dict(doc, counts=counts, seconds=round(time.time() - t0, 1)), indent=2))
        return 0 if hits else 1

    local = scan.to_eastern(now)
    print("Signal scan, %s ET, daily bars through %s." % (local.strftime("%a %Y-%m-%d %H:%M"),
                                                          last.isoformat() if last else "n/a"))
    print("Rule: %s." % rule)
    print("Fetched %s of %s charts; %s liquid ($%g+, %s+ shares); %d MACD candidates; %d checked with real OHLC%s."
          % ("{:,}".format(counts["fetched"]), "{:,}".format(len(symbols)), "{:,}".format(counts["liquid"]),
             a.min_price, "{:,}".format(a.min_volume), counts["macd_candidates"], counts["confirmed"],
             (" (%d failed)" % counts["stage2_failed"]) if counts["stage2_failed"] else ""))
    if hits:
        print("%d chart%s matched all three, most traded first:" % (len(hits), "" if len(hits) == 1 else "s"))
        print("  %-6s %10s  %10s  %-11s %9s" % ("symbol", "close", "supertrend", "MACD cross", "MACD"))
        for h in hits[:a.show]:
            ago = h["macd_cross_bars_ago"]
            print("  %-6s %10s  %10s  %-11s %9.3f" % (
                h["symbol"], scan.money(h["price"]), scan.money(h["supertrend"]),
                "today" if ago == 0 else "%d bar%s ago" % (ago, "" if ago == 1 else "s"), h["macd"]))
        if len(hits) > a.show:
            print("  ... and %d more in %s." % (len(hits) - a.show, a.out))
    else:
        print("No chart matched all three.")
    if not a.no_write:
        print("Wrote %s%s." % (a.out, (" and %s (TradingView import)" % a.tv_out) if a.tv_out else ""))
    print(NOTE + " Next: pick one chart, then /trader-news TICKER.")
    return 0 if hits else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Heikin Ashi + Supertrend + MACD signal scan on daily bars.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--symbols", help="comma-separated tickers instead of the universe")
    g.add_argument("--watchlist", metavar="FILE", help="a watchlist.json (hits) or a text file of tickers")
    ap.add_argument("--universe", help="symbol list (default %s)" % scan.DEFAULT_UNIVERSE)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--min-price", type=float, default=5.0)
    ap.add_argument("--min-volume", type=int, default=1000000)
    ap.add_argument("--cross-within", type=int, default=3, help="MACD cross on one of the last N bars (3)")
    ap.add_argument("--any-ha", action="store_true", help="do not require the candle before the two greens to be red")
    ap.add_argument("--st-length", type=int, default=10)
    ap.add_argument("--st-factor", type=float, default=3.0)
    ap.add_argument("--now", help='pretend it is this time ("2026-11-05 16:30" New York)')
    ap.add_argument("--out", default="trader/signals.json")
    ap.add_argument("--tv-out", default="trader/signals_tradingview.txt")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--pause", type=float, default=scan.PAUSE)
    ap.add_argument("--show", type=int, default=50)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    a.offline = None
    if a.cross_within < 1:
        ap.error("--cross-within must be 1 or more")
    try:
        return run(a)
    except scan.ScanError as e:
        print("signals.py: %s" % e, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("signals.py: stopped", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
