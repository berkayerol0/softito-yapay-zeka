import csv
import random
import time
import os

URLS = ["/index", "/login", "/logout", "/products", "/cart", "/checkout", "/profile", "/search", "/about", "/contact"]
METHODS = ["GET", "POST", "PUT", "DELETE"]
STATUS_CODES = [200, 200, 200, 200, 201, 304, 400, 401, 403, 404, 500, 502]
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)",
    "Mozilla/5.0 (Linux; Android 13; Pixel 7)",
    "curl/7.79.1",
    "Python-urllib/3.11",
    "Mozilla/5.0 (compatible; Googlebot/2.1)",
]
IPS = [f"192.168.{random.randint(0,255)}.{random.randint(1,254)}" for _ in range(1000)]
COUNTRIES = ["US", "DE", "GB", "FR", "BR", "IN", "JP", "CA", "AU", "TR"]

total_rows = 5_000_000
batch_size = 500_000
file_count = 10
rows_per_file = total_rows // file_count

os.makedirs("logs", exist_ok=True)

start_ts = int(time.time()) - 86400 * 7

for f_idx in range(file_count):
    filepath = f"logs/weblogs_{f_idx:03d}.csv"
    pc = f_idx + 1
    print(f"{pc}/{file_count} Uretiliyor: {filepath} ({rows_per_file} satir)")
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ip", "timestamp", "method", "url", "status", "response_time_ms", "user_agent", "country", "bytes_sent"])
        for _ in range(rows_per_file):
            ts = start_ts + random.randint(0, 86400 * 7)
            writer.writerow([
                random.choice(IPS),
                ts,
                random.choice(METHODS),
                random.choice(URLS),
                random.choice(STATUS_CODES),
                random.randint(1, 5000),
                random.choice(USER_AGENTS),
                random.choice(COUNTRIES),
                random.randint(100, 50000),
            ])
    size_mb = os.path.getsize(filepath) / 1024 / 1024
    print(f"   Tamam: {size_mb:.1f} MB")

print(f"\nToplam {total_rows} satir, {file_count} dosya olarak logs/ dizinine yazildi.")
