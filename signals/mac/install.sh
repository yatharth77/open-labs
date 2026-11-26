#!/bin/bash
# One-time setup on your Mac:  bash <(curl -fsSL https://raw.githubusercontent.com/yatharth77/open-labs/main/signals/mac/install.sh)
set -euo pipefail
REPO_URL="https://github.com/yatharth77/open-labs.git"
DIR="$HOME/open-labs-signals"
LABEL="com.openlabs.signals"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
STATE="$HOME/.openlabs-signals"

say() { printf "\n\033[1m%s\033[0m\n" "$*"; }

say "1/5 Checking Python 3 and git"
PY="$(command -v python3 || true)"
if [ -z "$PY" ] || ! "$PY" -c "import sys; assert sys.version_info >= (3, 9)" 2>/dev/null; then
  echo "Python 3.9+ not found. Run:  xcode-select --install   then re-run this installer."; exit 1
fi
command -v git >/dev/null || { echo "git not found. Run:  xcode-select --install"; exit 1; }
echo "python: $PY ($("$PY" --version))"

say "2/5 Getting the scanner (signals folder only - skips the 5 GB of minute data)"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" pull -q --rebase origin main
else
  git clone -q --depth 1 --filter=blob:none --sparse "$REPO_URL" "$DIR"
  git -C "$DIR" sparse-checkout set signals
fi
echo "installed in $DIR"

say "3/5 Checking you can push to GitHub (the dashboard updates through git)"
if ! git -C "$DIR" push --dry-run -q origin HEAD:main 2>/dev/null; then
  echo "Git can't push to yatharth77/open-labs from this Mac."
  echo "Fix it once with GitHub CLI:  brew install gh && gh auth login   (choose HTTPS, then 'Login with a web browser')"
  echo "Then re-run this installer."
  exit 1
fi
echo "push access OK"

say "4/5 Checking the price feed from this Mac"
"$PY" - <<'PYEOF'
import json, urllib.request, ssl, sys
url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?interval=1d&range=5d"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    j = json.load(urllib.request.urlopen(req, timeout=20))
    print("NIFTY last close:", round(j["chart"]["result"][0]["meta"]["regularMarketPrice"], 1), "- feed OK")
except ssl.SSLError as e:
    print("SSL certificate problem:", e)
    print("If you installed Python from python.org, run 'Install Certificates.command' in /Applications/Python 3.x/")
    sys.exit(1)
except Exception as e:
    print("Price feed check failed:", e); sys.exit(1)
PYEOF

say "5/5 Scheduling it (runs every minute; does real work only when a scan is due)"
mkdir -p "$STATE" "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>$PY</string><string>$DIR/signals/mac/agent.py</string></array>
  <key>StartInterval</key><integer>60</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$STATE/launchd.log</string>
  <key>StandardErrorPath</key><string>$STATE/launchd.log</string>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin</string></dict>
</dict></plist>
EOF
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "scheduled"

say "Done."
echo "Scans: 09:21, 11:00, 15:00 (+ scoring 15:46) on weekdays, and whenever you press Refresh."
echo "Your Mac must be awake then. To wake it automatically every weekday morning:"
echo "    sudo pmset repeat wakeorpoweron MTWRF 09:10:00"
echo "Logs:      tail -f $STATE/agent.log"
echo "Uninstall: bash $DIR/signals/mac/uninstall.sh"
