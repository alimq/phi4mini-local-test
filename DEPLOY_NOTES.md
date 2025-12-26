# Deploy notes

This bundle includes:
- Cursor-based incremental digest (`feed_into_phi.py` stores outbox byte offset in `/var/lib/phi4mini/seen.sqlite` key `outbox_offset_bytes`).
- Scraper scan improvement (`copied_scraper.py` scans deeper than `--max-items` to find NEW items; see `--scan-limit`).
- Continuous historical backfill beyond RSS via sitemaps (`backfill_discover.py` + `backfill_worker.py`) with `phi4mini-backfill.service`.

Extraction onto `/` will place files under:
- `/opt/phi4mini/*`
- `/etc/systemd/system/*`

Your data under `/var/lib/phi4mini` is not included.
