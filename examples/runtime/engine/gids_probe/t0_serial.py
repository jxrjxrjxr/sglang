"""T0: serial baseline -- GIDS used exactly as designed.

Each round: fill pages -> PUT -> EXIST -> GET -> verify bytes, one op at a
time on one thread. Nothing overlaps on the GPU, so this must be 100% clean.
If T0 fails, the problem is the environment/box, not concurrency; stop here.

    python t0_serial.py --rounds 20 --pages 32
"""

import argparse
import secrets
import sys

import torch

from gids_probe_lib import Gids, add_common_args, log, make_keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args()
    token = args.run_token or secrets.token_hex(8)
    n = args.pages
    gids = Gids(args.device, args.layers, n, args.kv_bytes, args.index_bytes, args.timeout)
    stream = torch.cuda.Stream(device=gids.device)
    slots = list(range(n))
    failures = 0
    for r in range(args.rounds):
        keys = make_keys(token, f"t0r{r}", n)
        before = gids.exist(keys, f"r{r} EXIST(before)")  # fresh keys: must be 0
        gids.fill(keys, slots, stream)
        gids.put(keys, slots, stream).wait(f"r{r} PUT")
        hit = gids.exist(keys, f"r{r} EXIST")
        gids.clear(slots, stream)
        if hit:
            gids.get(keys[:hit], slots[:hit], stream).wait(f"r{r} GET")
        check = gids.verify(keys[:hit], slots[:hit], stream)
        ok = before == 0 and hit == n and check["bad"] == 0
        failures += not ok
        log(
            f"[T0 r{r:02d}] exist_before={before} exist_after={hit}/{n} "
            f"data_bad={check['bad']} (untouched={check['untouched']}) "
            f"{'OK' if ok else 'FAIL bad_keys=' + str(check['bad_keys'])}"
        )
    verdict = "PASS" if failures == 0 else "FAIL"
    log(f"RESULT test=T0 verdict={verdict} failed_rounds={failures}/{args.rounds} token={token}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
