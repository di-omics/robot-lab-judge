"""A light, minimal contact sheet of last-frame top views with truth and verdict."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = (250, 250, 248)
INK = (60, 64, 70)
SOFT = (200, 202, 205)
WRONG = (200, 40, 40)
THUMB_W, THUMB_H = 224, 126
CAPTION_H = 34
PAD = 12
COLS = 6


def _font(size: int):
    for name in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/HelveticaNeue.ttc",
                 "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def contact_sheet(verdicts: list[dict], manifest: dict, out: Path) -> None:
    rows = sorted(verdicts, key=lambda r: (r["task"], r["seed"], r["take_id"]))
    n_rows = (len(rows) + COLS - 1) // COLS
    title_h = 40
    width = PAD + COLS * (THUMB_W + PAD)
    height = title_h + PAD + n_rows * (THUMB_H + CAPTION_H + PAD)
    sheet = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(sheet)
    small, big = _font(11), _font(14)
    wrong_n = sum(r["verdict"] != r["truth"].upper() for r in rows)
    d.text((PAD, 12), f"Last-frame top views, condition 'after'. Red frame = wrong verdict ({wrong_n} of {len(rows)}).",
           fill=INK, font=big)
    for i, r in enumerate(rows):
        x = PAD + (i % COLS) * (THUMB_W + PAD)
        y = title_h + PAD + (i // COLS) * (THUMB_H + CAPTION_H + PAD)
        thumb = Image.open(manifest[r["take_id"]]["images"]["last_top"]).convert("RGB")
        sheet.paste(thumb.resize((THUMB_W, THUMB_H), Image.LANCZOS), (x, y))
        wrong = r["verdict"] != r["truth"].upper()
        if wrong:
            d.rectangle((x - 2, y - 2, x + THUMB_W + 1, y + THUMB_H + 1), outline=WRONG, width=3)
        else:
            d.rectangle((x - 1, y - 1, x + THUMB_W, y + THUMB_H), outline=SOFT, width=1)
        d.text((x, y + THUMB_H + 4), f"{r['task']}  s{r['seed']}", fill=INK, font=small)
        mark = "  WRONG" if wrong else ""
        d.text((x, y + THUMB_H + 18), f"truth {r['truth'].upper()}  judged {r['verdict']}{mark}",
               fill=WRONG if wrong else INK, font=small)
    sheet.save(out, quality=90)
