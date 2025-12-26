# Phi4-mini digest web

This adds a tiny local web service that renders `/var/lib/phi4mini/digest.md` to readable HTML, and an nginx reverse proxy on ports **80/443**.

## Quick install (Ubuntu)

Run as root:

```bash
sudo /opt/phi4mini/setup_web.sh
```

Then visit:

- `http://<server-ip>/`
- `https://<server-ip>/` (will show a **browser warning** because the certificate is **self-signed**; without a domain you can't get a normal Let's Encrypt cert).

## Files

- Web server: `/opt/phi4mini/web/serve_digest.py`
- Systemd unit: `/etc/systemd/system/phi4mini-web.service`
- Nginx site: `/etc/nginx/sites-available/phi4mini-digest`

## Service commands

```bash
sudo systemctl status phi4mini-web
sudo journalctl -u phi4mini-web -f

sudo systemctl status nginx
sudo nginx -t
```

## Notes

- The web server listens on `127.0.0.1:8088` by default; nginx is what makes it public on 80/443.
- `GET /raw` returns the raw markdown.
- The HTML auto-refreshes every 60s (configurable via `PHI4MINI_DIGEST_REFRESH_SECS`).
