#!/bin/bash
# IR Thermography Dataset Downloader for CFRP Defect Detection
# Downloads 3 public datasets for FEM×IR fusion (Issue #3)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$(dirname "$SCRIPT_DIR")/raw"

echo "============================================"
echo "  IR Thermography Dataset Downloader"
echo "============================================"

# ---------- Dataset 1: Mendeley CFRP Defect Segmentation ----------
DS1_DIR="$RAW_DIR/mendeley_cfrp_seg"
echo ""
echo "[1/3] Mendeley - CFRP Defect Segmentation (Garcia Vargas & Fernandes 2025)"
echo "  URL: https://data.mendeley.com/datasets/jrsb4b9yy5/1"
echo "  Content: 1034 thermal images (640x512) + segmentation masks"
echo "  Material: CFRP (Carbon/PEEK), pulsed thermography"
echo ""
if [ -f "$DS1_DIR/.downloaded" ]; then
    echo "  [SKIP] Already downloaded."
else
    echo "  NOTE: Mendeley Data requires browser download."
    echo "  Steps:"
    echo "    1. Open: https://data.mendeley.com/datasets/jrsb4b9yy5/1"
    echo "    2. Click 'Download' (all files)"
    echo "    3. Extract to: $DS1_DIR/"
    echo "    4. Run: touch $DS1_DIR/.downloaded"
    echo ""
    echo "  Attempting API download..."
    # Mendeley public datasets can sometimes be downloaded via direct URL
    MENDELEY_URL1="https://data.mendeley.com/public-files/datasets/jrsb4b9yy5/files"
    if command -v wget &>/dev/null; then
        wget -q --spider "$MENDELEY_URL1" 2>/dev/null && \
            echo "  Direct download available - downloading..." && \
            wget -P "$DS1_DIR/" "$MENDELEY_URL1" 2>/dev/null || \
            echo "  Direct download not available. Please download manually."
    else
        echo "  wget not found. Please download manually."
    fi
fi

# ---------- Dataset 2: Mendeley Composite Academic Samples ----------
DS2_DIR="$RAW_DIR/mendeley_composite_samples"
echo ""
echo "[2/3] Mendeley - Composite Material Academic Samples"
echo "  URL: https://data.mendeley.com/datasets/v4knrwgj9y/2"
echo "  Content: 12 thermal sequences (~2000 frames each, 512x512)"
echo "  Material: 3 CFRP + 3 GFRP plates, pulsed thermography"
echo ""
if [ -f "$DS2_DIR/.downloaded" ]; then
    echo "  [SKIP] Already downloaded."
else
    echo "  NOTE: Mendeley Data requires browser download."
    echo "  Steps:"
    echo "    1. Open: https://data.mendeley.com/datasets/v4knrwgj9y/2"
    echo "    2. Click 'Download' (all files)"
    echo "    3. Extract to: $DS2_DIR/"
    echo "    4. Run: touch $DS2_DIR/.downloaded"
fi

# ---------- Dataset 3: Zenodo Step-Heating ----------
DS3_DIR="$RAW_DIR/zenodo_step_heating"
echo ""
echo "[3/3] Zenodo - Step-Heating Thermography CFRP (Pedrayes et al. 2022)"
echo "  URL: https://zenodo.org/records/5426793"
echo "  Content: 36 multi-band images (640x480, 30 channels each)"
echo "  Material: CFRP, step-heating thermography"
echo ""
if [ -f "$DS3_DIR/.downloaded" ]; then
    echo "  [SKIP] Already downloaded."
else
    echo "  Attempting Zenodo API download..."
    ZENODO_RECORD="5426793"
    # Zenodo provides direct download links via API
    if command -v wget &>/dev/null; then
        echo "  Fetching file list from Zenodo..."
        ZENODO_API="https://zenodo.org/api/records/$ZENODO_RECORD"
        # Try to download all files from the record
        python3 - "$ZENODO_API" "$DS3_DIR" <<'PYEOF'
import sys, json, urllib.request, os

api_url = sys.argv[1]
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

try:
    with urllib.request.urlopen(api_url, timeout=30) as resp:
        data = json.loads(resp.read())

    files = data.get("files", [])
    if not files:
        print("  No files found in Zenodo record.")
        sys.exit(1)

    for f in files:
        fname = f["key"]
        url = f["links"]["self"]
        size_mb = f["size"] / 1e6
        out_path = os.path.join(out_dir, fname)

        if os.path.exists(out_path):
            print(f"  [SKIP] {fname} already exists")
            continue

        print(f"  Downloading {fname} ({size_mb:.1f} MB)...")
        urllib.request.urlretrieve(url, out_path)
        print(f"  [OK] {fname}")

    # Mark as downloaded
    with open(os.path.join(out_dir, ".downloaded"), "w") as fp:
        fp.write("zenodo_step_heating downloaded\n")
    print("  [DONE] Zenodo dataset downloaded successfully.")

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Please download manually from: https://zenodo.org/records/{api_url.split('/')[-1]}")
    sys.exit(1)
PYEOF
    else
        echo "  wget/python3 not found. Please download manually from:"
        echo "    https://zenodo.org/records/$ZENODO_RECORD"
    fi
fi

echo ""
echo "============================================"
echo "  Download Summary"
echo "============================================"
for d in "$DS1_DIR" "$DS2_DIR" "$DS3_DIR"; do
    name=$(basename "$d")
    if [ -f "$d/.downloaded" ]; then
        echo "  [OK] $name"
    else
        echo "  [PENDING] $name - manual download required"
    fi
done
echo ""
echo "After downloading, run the conversion scripts:"
echo "  python3 $SCRIPT_DIR/convert_mendeley_cfrp_seg.py"
echo "  python3 $SCRIPT_DIR/convert_mendeley_composite.py"
echo "  python3 $SCRIPT_DIR/convert_zenodo_step_heating.py"
echo ""
echo "Or run the master pipeline:"
echo "  python3 $SCRIPT_DIR/run_ir_pipeline.py"
