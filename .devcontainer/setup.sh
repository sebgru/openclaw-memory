#!/usr/bin/env bash
# .devcontainer/setup.sh — runs once inside the container after creation.
# Safe to re-run; all steps are idempotent.
set -euo pipefail

# ── System packages ───────────────────────────────────────────────────────────
echo "→ Installing system packages (vim, jq)…"
sudo apt-get update -qq && sudo apt-get install -y --no-install-recommends vim jq

# ── Python dev tools ──────────────────────────────────────────────────────────
# Install the same tools used by CI so local and CI environments match.
echo "→ Installing project with dev dependencies…"
cd /workspace
pip install --upgrade pip
pip install -e '.[dev]'

echo "→ Installing standalone dev tools…"
pip install ruff pytest pytest-cov mypy

# ── Git config ────────────────────────────────────────────────────────────────
# .gitconfig is staged at /tmp/host-gitconfig (bind-mounted read-only).
# Copy it so git can write to ~/.gitconfig freely (bind-mounted files can't
# be atomically replaced, which causes "Device or resource busy" errors).
echo "→ Configuring git…"
if [ -s /tmp/host-gitconfig ]; then
    cp /tmp/host-gitconfig ~/.gitconfig
    echo "  ~/.gitconfig installed."
else
    echo "  No host .gitconfig found — skipping."
fi

# ── SSH keys ──────────────────────────────────────────────────────────────────
# .ssh is staged at /tmp/host-ssh (bind-mounted read-only from the host).
# We copy it to ~/.ssh with the permissions SSH requires (700/600).
# Contributors without an .ssh directory simply skip this step.
echo "→ Configuring SSH…"
if [ -d /tmp/host-ssh ] && [ -n "$(ls -A /tmp/host-ssh 2>/dev/null)" ]; then
    mkdir -p ~/.ssh
    cp -rp /tmp/host-ssh/. ~/.ssh/
    chmod 700 ~/.ssh
    find ~/.ssh -type f -exec chmod 600 {} \;
    echo "  SSH keys installed."
else
    echo "  No SSH keys found on host — skipping."
fi

echo ""
echo "✅ Container setup complete."
echo "   Workspace : /workspace"
echo "   Python    : $(python --version)"
echo "   pip       : $(pip --version | cut -d' ' -f2)"
echo "   ruff      : $(ruff --version 2>&1 | head -1)"
echo "   pytest    : $(pytest --version 2>&1 | head -1)"
echo "   mypy      : $(mypy --version 2>&1 | head -1)"
