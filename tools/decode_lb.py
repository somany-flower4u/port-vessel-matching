"""Decode 10/19+ leaderboard scores into (Hangul errors, non-Hangul errors).

The test has 1,064 Hangul and 67,792 non-Hangul rows (from the probes:
1 - score is always an integer over 677,920), so
    (1 - score) * 677,920 = 446 * ko_err + 3 * en_err.
Usage:  python decode_lb.py SCORE [--ko K | --en E]
"""
from __future__ import annotations

import argparse

N_KO, N_EN, UNIT = 1064, 67792, 677920


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("score", type=float)
    ap.add_argument("--ko", type=int, help="known Hangul error count")
    ap.add_argument("--en", type=int, help="known non-Hangul error count")
    args = ap.parse_args()
    x = (1 - args.score) * UNIT
    print(f"(1 - score) * {UNIT} = {x:.3f}")
    X = round(x)
    if args.ko is not None:
        print("non-Hangul errors:", (X - 446 * args.ko) / 3)
    elif args.en is not None:
        print("Hangul errors:", (X - 3 * args.en) / 446)
    else:
        for k in range(0, N_KO + 1):
            e = X - 446 * k
            if e >= 0 and e % 3 == 0 and e // 3 <= N_EN:
                print(f"  ko_err {k:4d}  en_err {e // 3:6d}")


if __name__ == "__main__":
    main()
