#!/usr/bin/env python3
"""
Runs the signals scans on your Mac (Yahoo works from a home connection; it blocks
GitHub's servers). launchd starts this every minute; it decides what is due:

  09:21+  intraday scan (once a day)       11:00+  long-term scan
  15:00+  overnight scan                   15:46+  scoring run
  any time: a Refresh request from the dashboard (signals/data/refresh_request.json)

Each run commits signals/data and pushes, so the dashboard on GitHub Pages updates.
Logs: ~/.openlabs-signals/agent.log
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE = os.path.expanduser("~/.openlabs-signals")
os.makedirs(STATE, exist_ok=True)
LOG = os.path.join(STATE, "agent.log")
PY = sys.executable
GIT = "/usr/bin/git" if os.path.exists("/usr/bin/git") else "git"

# flag, command, earliest, latest (HHMM, IST). Commands: engine sections, "collect", "recent".
SCHEDULE = [("intraday", "intraday", 921, 1500), ("long", "long", 1100, 1530), ("overnight", "overnight", 1500, 1529),
            ("score", "score", 1546, 2359), ("close", "long", 1548, 2359),   # post-close: "buy at next open" list
            ("collect", "collect", 1550, 2359)]


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{datetime.now(IST):%Y-%m-%d %H:%M:%S} {msg}\n")
    # keep the log small
    try:
        if os.path.getsize(LOG) > 2_000_000:
            lines = open(LOG).readlines()[-5000:]
            open(LOG, "w").writelines(lines)
    except OSError:
        pass


def git(*args, check=True):
    r = subprocess.run([GIT, *args], cwd=REPO, capture_output=True, text=True, timeout=300)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[:300]}")
    return r


def pull():
    r = git("pull", "-q", "--rebase", "-X", "theirs", "origin", "main", check=False)
    if r.returncode != 0:
        git("rebase", "--abort", check=False)
        log(f"pull failed: {r.stderr.strip()[:200]}")


def push(label, paths=("signals/data",)):
    git("add", *paths)
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        return
    git("commit", "-q", "-m", f"signals: {label} {datetime.now(IST):%Y-%m-%d %H:%M} IST (mac)")
    for _ in range(4):
        pull()
        if git("push", "-q", "origin", "HEAD:main", check=False).returncode == 0:
            log(f"pushed {label}")
            return
        time.sleep(5)
    log("push failed - will retry next minute")


def ensure_sparse():
    """the minute-data folder must be part of this sparse checkout to commit into it"""
    r = git("sparse-checkout", "list", check=False)
    if r.returncode == 0 and "nifty-data-live" not in r.stdout:
        git("sparse-checkout", "add", "nifty-data-live", check=False)


def run(cmd):
    log(f"run {cmd}")
    env = dict(os.environ, SIGNALS_WORKERS="4", SIGNALS_RUNNER="mac")
    if cmd == "collect":
        args, paths = [os.path.join(REPO, "signals", "collect.py")], ["nifty-data-live"]
    elif cmd == "backfill":
        args, paths = [os.path.join(REPO, "signals", "collect.py"), "--backfill"], ["nifty-data-live"]
    elif cmd == "recent":
        args, paths = [os.path.join(REPO, "signals", "recent.py")], ["signals/data"]
    else:
        args, paths = [os.path.join(REPO, "signals", "engine.py"), "--section", cmd], ["signals/data"]
    if "nifty-data-live" in paths:
        ensure_sparse()
    r = subprocess.run([PY, *args], cwd=REPO, capture_output=True, text=True, timeout=3600, env=env)
    log((r.stdout + r.stderr).strip()[-600:])
    push(cmd, paths)
    return r.returncode == 0


def main():
    lock = os.path.join(STATE, "lock")
    try:
        os.mkdir(lock)
    except FileExistsError:
        if time.time() - os.path.getmtime(lock) < 1800:
            return  # a run is already in progress
        os.rmdir(lock)
        os.mkdir(lock)
    try:
        now = datetime.now(IST)
        hm = now.hour * 100 + now.minute
        weekday = now.weekday() < 5
        if not (weekday and 800 <= hm <= 2300) and now.minute % 15:
            return  # off hours: only check for Refresh requests every 15 minutes
        pull()
        # 1) Refresh button
        req_path = os.path.join(REPO, "signals", "data", "refresh_request.json")
        seen_path = os.path.join(STATE, "last_refresh")
        if os.path.exists(req_path):
            try:
                req = json.load(open(req_path))
                rid = req.get("requested_at", "")
                seen = open(seen_path).read().strip() if os.path.exists(seen_path) else ""
                if rid and rid != seen:
                    open(seen_path, "w").write(rid)
                    run(req.get("section") or "all")
            except Exception as e:
                log(f"refresh request unreadable: {e}")
        # 2) schedule
        if weekday:
            day = now.strftime("%Y-%m-%d")
            for name, cmd, start, end in SCHEDULE:
                flag = os.path.join(STATE, f"done-{day}-{name}")
                if start <= hm <= end and not os.path.exists(flag):
                    if run(cmd):
                        open(flag, "w").close()
        # one-time backfill of the last ~30 days of minute data, then a recent-performance check
        if not os.path.exists(os.path.join(STATE, "backfilled")) and (not weekday or hm >= 1555 or hm < 900):
            if run("backfill"):
                open(os.path.join(STATE, "backfilled"), "w").close()
                run("recent")
        # monthly recent-performance check (first run of each month, after the close)
        month_flag = os.path.join(STATE, f"recent-{now:%Y-%m}")
        if os.path.exists(os.path.join(STATE, "backfilled")) and not os.path.exists(month_flag) \
                and (not weekday or hm >= 1610 or hm < 900):
            if run("recent"):
                open(month_flag, "w").close()
            # tidy old flags
            for f in os.listdir(STATE):
                if f.startswith("done-") and f[5:15] < (now - timedelta(days=7)).strftime("%Y-%m-%d"):
                    os.remove(os.path.join(STATE, f))
    except Exception as e:
        log(f"error: {e}")
    finally:
        try:
            os.rmdir(lock)
        except OSError:
            pass


if __name__ == "__main__":
    main()
