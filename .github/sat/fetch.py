#!/usr/bin/env python3
"""Fetch Meteosat imagery and cloud forecasts around Gardos (Belgrade) for an astro session."""
import io
import json
import math
import os
import re
import sys
import traceback
import datetime as dt
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import requests
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(OUT, exist_ok=True)
WMS = "https://view.eumetsat.int/geoserver/ows"
LAT, LON = 44.848, 20.410
TZ = ZoneInfo("Europe/Belgrade")
UTC = dt.timezone.utc
NOW = dt.datetime.now(UTC)
REGIONS = {
    "local": ((11.0, 40.5, 28.0, 49.5), 1020, 540),
    "wide": ((-6.0, 34.0, 38.0, 58.0), 1100, 600),
}
CATS = {
    "ir": r"ir_?10[58]|ir_?10\.[58]",
    "nightmicro": r"night.?micro|nmp",
    "wv": r"wv_?0?6[23]|wv_?0?73",
    "airmass": r"airmass",
    "dust": r"dust",
    "cloudtype": r"cloud.?type|ctype",
    "cth": r"cloud.?top|ctth|cth",
    "geocolour": r"geocolou?r",
}
WIDE_CATS = {"ir", "nightmicro", "wv", "airmass"}
ANIM_CATS = ["ir", "nightmicro", "wv"]

S = requests.Session()
S.headers["User-Agent"] = "astro-cloud-check/1.0 (GitHub Actions)"
summary = {"generated_utc": NOW.isoformat(), "errors": [], "candidates": {}, "chosen": {},
           "images": [], "ir_at_gardos": []}


def err(msg):
    print("ERROR:", msg, flush=True)
    summary["errors"].append(msg)


def font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


F = font(16)
FS = font(12)


def tag(el):
    return el.tag.rsplit("}", 1)[-1]


def parse_caps(xml_bytes):
    layers = {}
    for el in ET.fromstring(xml_bytes).iter():
        if tag(el) != "Layer":
            continue
        name = title = tspec = None
        for ch in el:
            t = tag(ch)
            if t == "Name":
                name = ch.text
            elif t == "Title":
                title = ch.text
            elif t in ("Dimension", "Extent") and ch.get("name") == "time" and (ch.text or "").strip():
                tspec = ch.text.strip()
        if name:
            layers[name] = {"title": title, "time": tspec}
    return layers


def parse_t(s):
    s = s.strip()
    if s.upper() in ("PRESENT", "CURRENT"):
        return NOW
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def times_of(spec, hours=6):
    out = set()
    if not spec:
        return []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "/" in part:
                a, b, p = part.split("/")
                m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", p)
                step = dt.timedelta(hours=int(m[1] or 0), minutes=int(m[2] or 0), seconds=int(m[3] or 0))
                if step.total_seconds() <= 0:
                    continue
                start, end = parse_t(a), min(parse_t(b), NOW)
                t = end
                while t >= start and t >= end - dt.timedelta(hours=hours):
                    out.add(t)
                    t -= step
            else:
                t = parse_t(part)
                if NOW - dt.timedelta(hours=hours) <= t <= NOW + dt.timedelta(minutes=5):
                    out.add(t)
        except Exception:
            pass
    return sorted(out)


def pick_frames(times, span_h=3, every_min=30):
    if not times:
        return []
    latest = times[-1]
    picked = []
    for t in reversed(times):
        if t < latest - dt.timedelta(hours=span_h, minutes=1):
            break
        if not picked or picked[-1] - t >= dt.timedelta(minutes=every_min - 2):
            picked.append(t)
    return list(reversed(picked))


def tstr(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def getmap(layer, bbox, w, h, t=None, transparent=False):
    p = {"service": "WMS", "version": "1.1.1", "request": "GetMap", "layers": layer, "styles": "",
         "srs": "EPSG:4326", "bbox": ",".join(str(v) for v in bbox), "width": w, "height": h,
         "format": "image/png"}
    if transparent:
        p["transparent"] = "true"
    if t is not None:
        p["time"] = tstr(t)
    r = S.get(WMS, params=p, timeout=120)
    if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image"):
        raise RuntimeError(f"GetMap {layer} {p.get('time')}: HTTP {r.status_code} "
                           f"{r.headers.get('content-type')} {r.text[:300]!r}")
    return Image.open(io.BytesIO(r.content)).convert("RGBA")


def featureinfo(layer, t):
    d = 0.05
    p = {"service": "WMS", "version": "1.1.1", "request": "GetFeatureInfo", "layers": layer,
         "query_layers": layer, "styles": "", "srs": "EPSG:4326",
         "bbox": f"{LON - d},{LAT - d},{LON + d},{LAT + d}", "width": 11, "height": 11, "x": 5, "y": 5,
         "info_format": "application/json", "feature_count": 1, "format": "image/png"}
    if t is not None:
        p["time"] = tstr(t)
    r = S.get(WMS, params=p, timeout=60)
    try:
        j = r.json()
        feats = j.get("features") or []
        return feats[0].get("properties") if feats else j
    except ValueError:
        return r.text[:200]


def decorate(img, bbox, overlays):
    for ov in overlays:
        if ov.size == img.size:
            img = Image.alpha_composite(img, ov)
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = bbox
    w, h = img.size

    def px(lon, lat):
        return (lon - x0) / (x1 - x0) * w, (y1 - lat) / (y1 - y0) * h

    step = 5 if x1 - x0 > 20 else 2
    lon = math.ceil(x0 / step) * step
    while lon <= x1:
        x, _ = px(lon, 0)
        d.line([(x, 0), (x, h)], fill=(255, 255, 0, 110), width=1)
        d.text((x + 2, h - 14), f"{lon}E", fill=(255, 255, 0, 255), font=FS)
        lon += step
    lat = math.ceil(y0 / step) * step
    while lat <= y1:
        _, y = px(0, lat)
        d.line([(0, y), (w, y)], fill=(255, 255, 0, 110), width=1)
        d.text((2, y + 2), f"{lat}N", fill=(255, 255, 0, 255), font=FS)
        lat += step
    # range rings around Gardos: 100/200/300 km
    for km in (100, 200, 300):
        dlat = km / 111.2
        dlon = km / (111.2 * math.cos(math.radians(LAT)))
        ax, ay = px(LON - dlon, LAT + dlat)
        bx, by = px(LON + dlon, LAT - dlat)
        d.ellipse([ax, ay, bx, by], outline=(0, 255, 255, 200), width=1)
    cx, cy = px(LON, LAT)
    d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=(255, 0, 0, 255), width=2)
    d.line([(cx - 11, cy), (cx + 11, cy)], fill=(255, 0, 0, 255), width=1)
    d.line([(cx, cy - 11), (cx, cy + 11)], fill=(255, 0, 0, 255), width=1)
    return img


def label(img, text, f=F):
    d = ImageDraw.Draw(img)
    box = d.textbbox((0, 0), text, font=f)
    d.rectangle([0, 0, box[2] + 8, box[3] + 6], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255), font=f)
    return img


def when(t):
    return f"{t:%H:%M} UTC / {t.astimezone(TZ):%H:%M} local"


def save(img, name):
    path = os.path.join(OUT, name)
    img.convert("RGB").save(path, quality=88)
    summary["images"].append(name)
    print("saved", name, flush=True)


def sheet(frames, cols=3, tw=560):
    th = int(frames[0][1].height * tw / frames[0][1].width)
    rows = (len(frames) + cols - 1) // cols
    out = Image.new("RGB", (cols * tw, rows * th), "black")
    for i, (t, f) in enumerate(frames):
        tile = label(f.convert("RGB").resize((tw, th), Image.LANCZOS), when(t), FS)
        out.paste(tile, ((i % cols) * tw, (i // cols) * th))
    return out


def satellite():
    caps = S.get(WMS, params={"service": "WMS", "version": "1.3.0", "request": "GetCapabilities"}, timeout=240)
    caps.raise_for_status()
    layers = parse_caps(caps.content)
    with open(os.path.join(OUT, "layers.txt"), "w") as f:
        for n, v in sorted(layers.items()):
            spec = v["time"] or ""
            f.write(f"{n}\t{v['title']}\t{spec[:60]} ... {spec[-60:]}\n")
    print(f"{len(layers)} layers in capabilities", flush=True)

    def ws_rank(n):
        ws = n.split(":")[0].lower()
        if ws.startswith("mtg"):
            return 0
        return {"msg_rss": 1, "msg_fes": 2}.get(ws, 9)

    sat = [n for n in layers if ws_rank(n) < 9 and "iodc" not in n.lower()]
    chosen = {}
    for cat, rx in CATS.items():
        cands = sorted((n for n in sat if re.search(rx, n.split(":", 1)[-1], re.I)), key=lambda n: (ws_rank(n), n))
        summary["candidates"][cat] = cands
        picked, seen_ws = [], set()
        for n in cands:
            ws = n.split(":")[0]
            if ws not in seen_ws:
                picked.append(n)
                seen_ws.add(ws)
            if len(picked) == 2:
                break
        chosen[cat] = picked
    summary["chosen"] = chosen

    ovl_names = [n for n in layers if re.search(r"coast|boundar|border", n, re.I)]
    summary["candidates"]["overlay"] = ovl_names
    pick_ovl = []
    for rx in (r"coast", r"boundar|border"):
        for n in sorted(ovl_names, key=lambda n: (0 if n.lower().startswith(("backgrounds", "overlay")) else 1, len(n))):
            if re.search(rx, n, re.I):
                pick_ovl.append(n)
                break
    overlays = {}
    for reg, (bbox, w, h) in REGIONS.items():
        overlays[reg] = []
        for n in pick_ovl:
            try:
                overlays[reg].append(getmap(n, bbox, w, h, transparent=True))
            except Exception as e:
                err(f"overlay {n}: {e}")

    # latest image per chosen layer
    for cat, names in chosen.items():
        for n in names:
            times = times_of(layers[n]["time"])
            t = times[-1] if times else None
            for reg, (bbox, w, h) in REGIONS.items():
                if reg == "wide" and cat not in WIDE_CATS:
                    continue
                try:
                    img = decorate(getmap(n, bbox, w, h, t), bbox, overlays[reg])
                    label(img, f"{n}  {when(t) if t else 'latest'}")
                    save(img, f"latest_{reg}_{cat}_{n.replace(':', '_')}.jpg")
                except Exception as e:
                    err(f"latest {n} {reg}: {e}")

    # 3 h of frames for motion
    bbox, w, h = REGIONS["local"]
    for cat in ANIM_CATS:
        if not chosen.get(cat):
            continue
        n = chosen[cat][0]
        frames = []
        for t in pick_frames(times_of(layers[n]["time"])):
            try:
                frames.append((t, decorate(getmap(n, bbox, w, h, t), bbox, overlays["local"])))
            except Exception as e:
                err(f"frame {n} {t}: {e}")
            if cat == "ir":
                try:
                    summary["ir_at_gardos"].append({"time": tstr(t), "layer": n, "value": featureinfo(n, t)})
                except Exception as e:
                    err(f"featureinfo {n} {t}: {e}")
        if frames:
            save(sheet(frames), f"anim_{cat}_{n.replace(':', '_')}.jpg")
            save(label(frames[-2][1] if len(frames) > 1 else frames[-1][1], f"{n}  {when(frames[-2][0] if len(frames) > 1 else frames[-1][0])}"),
                 f"prev_local_{cat}.jpg")


OM = "https://api.open-meteo.com/v1/forecast"
HOURLY = ("cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,relative_humidity_2m,temperature_2m,"
          "dew_point_2m,visibility,wind_speed_10m,wind_direction_10m,wind_speed_300hPa,wind_direction_300hPa,"
          "relative_humidity_300hPa,wind_speed_250hPa,wind_direction_250hPa")
MODELS = ["best_match", "icon_eu", "icon_global", "ecmwf_ifs025", "gfs_seamless", "meteofrance_arpege_europe",
          "ukmo_global_deterministic_10km", "gem_global"]


def forecasts():
    lines, raw = [], {}
    for m in MODELS:
        try:
            r = S.get(OM, params={"latitude": LAT, "longitude": LON, "hourly": HOURLY, "models": m,
                                  "timezone": "Europe/Belgrade", "past_hours": 3, "forecast_hours": 12}, timeout=60)
            j = r.json()
            raw[m] = j
            if r.status_code != 200:
                err(f"open-meteo {m}: {r.status_code} {str(j)[:200]}")
                continue
        except Exception as e:
            err(f"open-meteo {m}: {e}")
            continue
        hr = j.get("hourly", {})

        def g(k, i, fmt="{:>4}"):
            v = hr.get(k)
            return fmt.format("-") if not v or v[i] is None else fmt.format(round(v[i]))

        lines.append(f"== {m} (local time) ==")
        lines.append("time   tot  low  mid high  rh2  t2  td2  vis_km  w300(dir/kmh) rh300")
        for i, t in enumerate(hr.get("time", [])):
            vis = hr.get("visibility")
            vkm = "-" if not vis or vis[i] is None else f"{vis[i] / 1000:.0f}"
            lines.append(f"{t[11:16]}  {g('cloud_cover', i)} {g('cloud_cover_low', i)} {g('cloud_cover_mid', i)} "
                         f"{g('cloud_cover_high', i)} {g('relative_humidity_2m', i)} {g('temperature_2m', i, '{:>3}')} "
                         f"{g('dew_point_2m', i, '{:>3}')}  {vkm:>6}  {g('wind_direction_300hPa', i)}/{g('wind_speed_300hPa', i, '{:<4}')}"
                         f"      {g('relative_humidity_300hPa', i)}")
        lines.append("")
    with open(os.path.join(OUT, "openmeteo.json"), "w") as f:
        json.dump(raw, f)
    with open(os.path.join(OUT, "forecast.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines), flush=True)

    for url in ("https://www.7timer.info/bin/astro.php", "http://www.7timer.info/bin/astro.php"):
        try:
            r = S.get(url, params={"lon": LON, "lat": LAT, "ac": 0, "unit": "metric", "output": "json",
                                   "tzshift": 0}, timeout=60)
            with open(os.path.join(OUT, "7timer.json"), "w") as f:
                f.write(r.text)
            print("7timer:", r.text[:1500], flush=True)
            break
        except Exception as e:
            err(f"7timer {url}: {e}")


def astro():
    import ephem
    obs = ephem.Observer()
    obs.lat, obs.lon, obs.elevation = str(LAT), str(LON), 120
    veil = ephem.FixedBody()
    veil._ra, veil._dec, veil._epoch = ephem.hours("20:51:00"), ephem.degrees("30:40:00"), ephem.J2000
    moon, sun = ephem.Moon(), ephem.Sun()
    lines = ["local  veil_alt veil_az  moon_alt moon_az moon_illum% moon-veil_sep sun_alt"]
    for k in range(0, 13):
        t = NOW + dt.timedelta(minutes=30 * k)
        obs.date = ephem.Date(t.replace(tzinfo=None))
        for b in (veil, moon, sun):
            b.compute(obs)
        sep = math.degrees(ephem.separation(veil, moon))
        lines.append(f"{t.astimezone(TZ):%H:%M}  {math.degrees(veil.alt):6.1f} {math.degrees(veil.az):7.1f}  "
                     f"{math.degrees(moon.alt):7.1f} {math.degrees(moon.az):7.1f}  {moon.phase:9.0f}  "
                     f"{sep:10.0f}  {math.degrees(sun.alt):6.1f}")
    obs.date = ephem.Date(NOW.replace(tzinfo=None))
    ms = ephem.Date(obs.next_setting(ephem.Moon())).datetime().replace(tzinfo=UTC)
    lines.append(f"moonset: {ms.astimezone(TZ):%Y-%m-%d %H:%M} local")
    with open(os.path.join(OUT, "astro.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines), flush=True)


for step in (forecasts, astro, satellite):
    try:
        step()
    except Exception:
        err(f"{step.__name__}: {traceback.format_exc()}")

with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=1, default=str)
print(json.dumps(summary, indent=1, default=str)[:6000], flush=True)
