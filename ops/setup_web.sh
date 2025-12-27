#!/usr/bin/env bash
set -euo pipefail

# Installs a small web UI for /var/lib/phi4mini/digest.md on ports 80/443.
# Safe to run multiple times.

WEB_DIR="/opt/phi4mini/app/web"
VENV_DIR="$WEB_DIR/.venv"
NGINX_SITE_AVAIL="/etc/nginx/sites-available/phi4mini-digest"
NGINX_SITE_ENABLED="/etc/nginx/sites-enabled/phi4mini-digest"
CERT_DIR="/etc/ssl/phi4mini-digest"

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (sudo)." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y nginx openssl python3 python3-venv

# Create venv + install python deps
mkdir -p "$WEB_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$WEB_DIR/requirements-web.txt"

# Self-signed cert for HTTPS on bare IP (browser will warn).
mkdir -p "$CERT_DIR"
if [[ ! -f "$CERT_DIR/fullchain.pem" || ! -f "$CERT_DIR/privkey.pem" ]]; then
  openssl req -x509 -nodes -newkey rsa:2048 -days 3650 \
    -keyout "$CERT_DIR/privkey.pem" \
    -out "$CERT_DIR/fullchain.pem" \
    -subj "/CN=phi4mini-digest" >/dev/null 2>&1
  chmod 600 "$CERT_DIR/privkey.pem"
fi

# Install nginx site file from the bundle (this script expects it to exist)
if [[ ! -f "$NGINX_SITE_AVAIL" ]]; then
  echo "Missing $NGINX_SITE_AVAIL. Did you extract the archive at / ?" >&2
  exit 1
fi

ln -sf "$NGINX_SITE_AVAIL" "$NGINX_SITE_ENABLED"
rm -f /etc/nginx/sites-enabled/default || true

# Reload systemd + enable services
systemctl daemon-reload
systemctl enable --now phi4mini-web.service

nginx -t
systemctl enable --now nginx
systemctl reload nginx

echo
echo "OK. Digest web is live:" 
echo "  http://$(hostname -I | awk '{print $1}')/"
echo "  https://$(hostname -I | awk '{print $1}')/  (self-signed cert -> browser warning)"
echo
echo "Logs:" 
echo "  journalctl -u phi4mini-web -f"
