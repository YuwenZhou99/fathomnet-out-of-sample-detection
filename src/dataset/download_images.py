import os, json, argparse, logging, time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fathomnet.api import images as fn_images  # pip install fathomnet

def build_session():
    s = requests.Session()
    retry = Retry(
        total=10, connect=10, read=10, status=10,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; FathomNetDownloader/1.0)"})
    return s

def uuid_from_filename(fn: str) -> str:
    return os.path.splitext(os.path.basename(fn))[0]

def download(session, url, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True
    tmp = out_path + ".part"
    with session.get(url, stream=True, timeout=(10, 120)) as r:
        if r.status_code >= 400:
            return False
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
    os.replace(tmp, out_path)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coco_json")
    ap.add_argument("--outdir", default="images")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    session = build_session()

    with open(args.coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    failures = 0
    for im in coco["images"]:
        fn = im["file_name"]                      # uuid.png
        uuid = uuid_from_filename(fn)
        try:
            rec = fn_images.find_by_uuid(uuid)    # has .url
            url = rec.url
        except Exception as e:
            failures += 1
            logging.warning(f"UUID lookup failed {uuid}: {e}")
            continue

        ok = download(session, url, os.path.join(args.outdir, fn))
        if not ok:
            failures += 1
            logging.warning(f"Download failed {uuid} -> {url}")

        time.sleep(0.05)  # 温和一点，避免触发限流

    logging.info(f"Done. failures={failures}")

if __name__ == "__main__":
    main()
