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

# ── Commit signing ────────────────────────────────────────────────────────────
# A host .gitconfig commonly enables SSH commit signing with
# `user.signingkey ~/.ssh/id_rsa.pub`. The private key normally lives in the
# forwarded ssh-agent rather than on disk, so that public key file ships from
# the host's .ssh directory only when the keypair is stored there too. Without
# it, ssh-keygen cannot resolve the key and every commit fails with
# "Couldn't load public key … No such file or directory". Materialize the
# public key from the agent so signing works either way.
echo "→ Configuring commit signing…"
if [ "$(git config --get gpg.format 2>/dev/null || true)" = "ssh" ]; then
    signing_key="$(git config --get user.signingkey 2>/dev/null || true)"
    signing_key="${signing_key/#\~/$HOME}"
    if [ -n "$signing_key" ] && [ ! -f "$signing_key" ]; then
        mkdir -p "$(dirname "$signing_key")"
        if ssh-add -L > "$signing_key" 2>/dev/null && [ -s "$signing_key" ]; then
            chmod 644 "$signing_key"
            echo "  Public key written from ssh-agent: $signing_key"
        else
            rm -f "$signing_key"
            echo "  WARNING: no keys in ssh-agent — commit signing will fail."
            echo "           Forward a key (ssh-add) or set commit.gpgsign=false."
        fi
    else
        echo "  Signing key already present or not configured — skipping."
    fi
else
    echo "  SSH commit signing not configured — skipping."
fi

echo ""
echo "✅ Container setup complete."
echo "   Workspace : /workspace"
echo "   Python    : $(python --version)"
echo "   pip       : $(pip --version | cut -d' ' -f2)"
echo "   ruff      : $(ruff --version 2>&1 | head -1)"
echo "   pytest    : $(pytest --version 2>&1 | head -1)"
echo "   mypy      : $(mypy --version 2>&1 | head -1)"
