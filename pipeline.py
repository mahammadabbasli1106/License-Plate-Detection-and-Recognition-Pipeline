"""
License plate pipeline: YOLOv8 detector -> padded crop -> OCR -> rule-based cleaning.

Main entry point:
    read_plates(image, model, reader=None, engine="easyocr") -> list of dicts
        box   : (x1, y1, x2, y2) from the MODEL's own output (no manual cropping)
        conf  : detector confidence
        crop  : padded plate crop (BGR numpy array)
        raw   : raw OCR text
        text  : cleaned plate text (Turkish format rules applied)
        valid : True if the cleaned text matches the Turkish plate regex
"""
import re
import cv2
import numpy as np

# ---------------------------------------------------------------- plate format
# Turkish plate = 2-digit province code (01-81) + letters + digits:
#   1 letter + 4 digits | 2 letters + 3-4 digits | 3 letters + 2-3 digits
PLATE_REGEX = re.compile(
    r"^(0[1-9]|[1-7][0-9]|8[01])([A-Z][0-9]{4}|[A-Z]{2}[0-9]{3,4}|[A-Z]{3}[0-9]{2,3})$"
)
ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Common OCR confusions, applied only where the plate format says which kind of
# character must be at that position.
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
            "S": "5", "B": "8", "G": "6"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A"}


def compact(s):
    """Uppercase and keep only A-Z / 0-9 (used for both predictions and ground truth)."""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def _fix(segment, table, want):
    """Force every char in segment to be a digit or a letter. Returns (text, n_substitutions)."""
    out, subs = [], 0
    for ch in segment:
        ok = ch.isdigit() if want == "digit" else ch.isalpha()
        if ok:
            out.append(ch)
        elif ch in table:
            out.append(table[ch])
            subs += 1
        else:
            return None, 99
    return "".join(out), subs


def clean_plate(raw):
    """
    Rule-based cleaning for Turkish plates.
    1. Strip spaces/symbols, uppercase.
    2. Drop the 'TR' country code printed on the blue EU strip.
    3. Positions 1-2 must be digits (O->0, I->1, B->8 ...).
    4. Try every legal letters/digits split; in the letter block fix 0->O, 1->I, 8->B ...,
       in the digit block fix O->0 etc. Keep the valid candidate needing the fewest fixes.
    Returns (clean_text, is_valid_format).
    """
    s = compact(raw)
    if s.startswith("TR") and len(s) > 7:
        s = s[2:]
    if len(s) < 6:
        return s, False

    province, p_subs = _fix(s[:2], TO_DIGIT, "digit")
    if province is None:
        return s, False

    best = None
    rest = s[2:]
    for n_letters in (1, 2, 3):
        letters, l_subs = _fix(rest[:n_letters], TO_LETTER, "letter")
        digits, d_subs = _fix(rest[n_letters:], TO_DIGIT, "digit")
        if letters is None or digits is None:
            continue
        cand = province + letters + digits
        if PLATE_REGEX.match(cand):
            score = p_subs + l_subs + d_subs
            if best is None or score < best[1]:
                best = (cand, score)
    if best:
        return best[0], True
    return s, False


def format_plate(s):
    """'06ABC123' -> '06 ABC 123' for display."""
    m = re.match(r"^(\d{2})([A-Z]+)(\d+)$", s)
    return f"{m.group(1)} {m.group(2)} {m.group(3)}" if m else s


# ---------------------------------------------------------------- detection
def detect_plates(img, model, conf=0.25):
    """Run the YOLO model. Returns [(box_xyxy, conf), ...] sorted by confidence. [] if none."""
    r = model.predict(img, conf=conf, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    cf = r.boxes.conf.cpu().numpy()
    dets = [(tuple(int(v) for v in b), float(c)) for b, c in zip(xyxy, cf)]
    return sorted(dets, key=lambda d: -d[1])


def crop_with_padding(img, box, pad=0.10):
    """Crop box + pad% on each side, clamped to image borders."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    return img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


# ---------------------------------------------------------------- OCR
def preprocess(crop):
    """Grayscale, upscale small plates to ~100px tall, denoise while keeping edges."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scale = max(1.0, 100 / max(g.shape[0], 1))
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.bilateralFilter(g, 9, 75, 75)


def run_ocr(crop, engine="easyocr", reader=None):
    g = preprocess(crop)
    if engine == "easyocr":
        res = reader.readtext(g, allowlist=ALNUM, detail=1, paragraph=False)
        res = sorted(res, key=lambda r: min(p[0] for p in r[0]))  # left -> right
        return " ".join(r[1] for r in res)
    if engine == "tesseract":
        import pytesseract
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cfg = f"--psm 7 -c tessedit_char_whitelist={ALNUM}"
        return pytesseract.image_to_string(b, config=cfg).strip()
    raise ValueError(f"unknown engine {engine}")


# ---------------------------------------------------------------- pipeline
def read_plates(image, model, reader=None, engine="easyocr", conf=0.25, pad=0.10):
    """Image (path or BGR array) in -> list of plate results out. Handles 0 or many plates."""
    img = cv2.imread(image) if isinstance(image, str) else image
    if img is None:
        raise ValueError(f"could not read image: {image}")
    results = []
    for box, c in detect_plates(img, model, conf):
        crop, _ = crop_with_padding(img, box, pad)
        if crop.size == 0:
            continue
        raw = run_ocr(crop, engine, reader)
        text, valid = clean_plate(raw)
        results.append(dict(box=box, conf=c, crop=crop, raw=raw, text=text, valid=valid))
    return results


def draw_results(img, results, gt_boxes=None):
    out = img.copy()
    for g in gt_boxes or []:
        cv2.rectangle(out, g[:2], g[2:], (0, 200, 0), 2)  # green = ground truth
    for r in results:
        x1, y1, x2, y2 = r["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red = prediction
        label = f"{format_plate(r['text'])} {r['conf']:.2f}"
        cv2.putText(out, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
    return out


# ---------------------------------------------------------------- metrics
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred, gt):
    """Character error rate = edit distance / length of ground truth."""
    gt = compact(gt)
    return levenshtein(compact(pred), gt) / max(len(gt), 1)
