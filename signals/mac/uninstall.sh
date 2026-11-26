#!/bin/bash
# Stops the Mac scanner and removes its files.
LABEL="com.openlabs.signals"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/$LABEL.plist"
rm -rf "$HOME/.openlabs-signals"
echo "Scanner stopped. The code is still in ~/open-labs-signals - delete that folder too if you like."
