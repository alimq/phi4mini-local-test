#!/usr/bin/env bash
set -euo pipefail

# Installs / enables:
#  - phi4mini-summarizer.service (existing scrape+digest loop, now also indexes RAG during the old sleep window)
#  - phi4mini-summarizer-web.service (digest viewer)
#  - phi4mini-rag.service (global RAG over collected data)
#  - phi4mini-rag-user.service (RAG over uploaded user documents)
#  - nginx reverse proxy routes:
#      /summarizer/ -> 8001
#      /rag/        -> 8002
#      /rag-user/   -> 8003

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (sudo)." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update -y
apt-get install -y nginx openssl python3 python3-venv

# Python venv for all services
if [[ ! -x /opt/phi4mini/.venv/bin/python ]]; then
  python3 -m venv /opt/phi4mini/.venv
fi

/opt/phi4mini/.venv/bin/python -m pip install --upgrade pip
/opt/phi4mini/.venv/bin/python -m pip install -r /opt/phi4mini/requirements.txt

# Self-signed SSL cert (optional; browsers will warn on IP)
CERT_DIR=/etc/ssl/phi4mini
mkdir -p "$CERT_DIR"
if [[ ! -f "$CERT_DIR/fullchain.pem" || ! -f "$CERT_DIR/privkey.pem" ]]; then
  openssl req -x509 -nodes -newkey rsa:2048 -days 3650 \
    -keyout "$CERT_DIR/privkey.pem" \
    -out "$CERT_DIR/fullchain.pem" \
    -subj "/CN=phi4mini" >/dev/null 2>&1
  chmod 600 "$CERT_DIR/privkey.pem"
fi

# This archive is meant to be extracted at / so that the unit files land at:
#   /etc/systemd/system/phi4mini-*.service
# If you extracted elsewhere, copy them there before running this script.

systemctl daemon-reload
systemctl enable --now phi4mini-summarizer.service
systemctl enable --now phi4mini-summarizer-web.service
systemctl enable --now phi4mini-rag.service
systemctl enable --now phi4mini-rag-user.service

# nginx site
if [[ -f /etc/nginx/sites-available/phi4mini ]]; then
  ln -sf /etc/nginx/sites-available/phi4mini /etc/nginx/sites-enabled/phi4mini
  rm -f /etc/nginx/sites-enabled/default || true
fi

nginx -t
systemctl enable --now nginx
systemctl reload nginx

echo
IP=$(hostname -I | awk '{print $1}')
echo "OK. Sites are live (self-signed SSL will warn in browsers):"
echo "  http://$IP/summarizer/"
echo "  http://$IP/rag/"
echo "  http://$IP/rag-user/"
echo "  https://$IP/summarizer/"
echo
echo "Logs:"
echo "  journalctl -u phi4mini-summarizer -f"
echo "  journalctl -u phi4mini-rag -f"
echo "  journalctl -u phi4mini-rag-user -f"
