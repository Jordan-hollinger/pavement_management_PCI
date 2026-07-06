"""
Author: Christina Herrick
Org:    Merrimack Valley Planning Commission
Email:  cherrick@mvpc.org
Phone:  978-374-0519

Calculate ASTM D6433 Pavement Condition Index (PCI) from any CSV by
interactively mapping its columns to the required fields.

Usage from CLI:
    python calculate_PCI_simplified.py input.csv [--output OUTPUT]

# =============================================================================
# INPUT CSV FORMAT
# =============================================================================
#
# The script reads the entire CSV as strings (all columns as text) and does
# its own numeric conversion internally, so numeric columns do not need to be
# pre-formatted in any particular way.
#
# One row = one distress observation. If your data has already been aggregated
# (one row per segment/class/severity combination), that is fine — the script
# will sum within each group, which is a no-op on pre-aggregated data.
#
# Your csv must have a column that contains: distress type, severity, and a unique id 
# for your road/road segment. It must also have a column that lists the extent of the 
# distress in percent (0 - 100) OR two columns with the physical extent of the distress 
# (ft/sqft) and the total length/area (ft/sqft) of the road/road segment.
#
# There is no required row order and no required column order.
# A header row is required (pandas reads the first row as column names).
#
# Rows are dropped and excluded from PCI if any of the following are true:
#   - The distress class value is mapped to "skip"
#   - The severity value is mapped to "skip"
#   - The rd_pct value is missing, blank, or non-numeric
#   - The rd_pct denominator is zero (Option 2 extent computation only)
#

# =============================================================================
# USER GUIDE — Input Requirements
# =============================================================================
#
# COLUMN NAMES
#   Your CSV column names can be anything. The script prompts you to pick
#   which column serves each role. However, the VALUES inside those columns
#   must match (or be manually mapped to) the labels listed below.
#
# -- DISTRESS CLASS VALUES
#   Your class column must contain values mappable to one of these five labels:
#     single-crack    — longitudinal, transverse, or diagonal cracks
#     crack-seal      — crack sealing / filled cracks
#     alligator-crack — alligator / fatigue cracking
#     patch           — patching and utility cut patching
#     pothole         — potholes
#
#   If your CSV uses different names (e.g. "Alligator Cracking", "D20"), the
#   script will prompt you to map each unique value to one of the five labels,
#   or to skip rows with that value entirely.
#
# -- SEVERITY VALUES
#   Your severity column must contain values mappable to exactly:
#     low  |  moderate  |  high
#
#   Exact case-insensitive matches are accepted automatically. Anything else
#   (e.g. "L", "Med", "3") will prompt a manual mapping or skip.
#
# DISTRESS EXTENT (rd_pct)
#   Option 1 — existing percentage column:
#     A column already expressing distress extent as % of road area or length
#     (0–100). Values above 100 are clipped to 100 automatically.
#   Option 2 — compute from two columns:
#     Pick a distress extent column and a road total column; the script
#     computes (extent / total * 100). Zero or non-numeric denominators
#     produce NaN and those rows are dropped.
#
#   Any other columns can optionally be passed through to outputs when
#   prompted — they do not affect PCI calculations.
#
# REQUIRED LOOKUP TABLE FILES (must be in the same directory as this script)
#   distress_alligator_area.csv
#   distress_crack_length.csv
#   distress_patching_area.csv
#   distress_pothole_area.csv
#   astm_cdv_correction_table.csv
#
# OUTPUTS (written next to input CSV, or to --output directory)
#   *_DV.csv     — aggregated rows with deduct value (DV) column added
#   *_segPCI.csv — DV rows joined with each segment's PCI result
#   *_allPCI.csv — one row per segment: pci, rating, max_cdv, cdv_a, cdv_b, method
#
# =============================================================================

You will be prompted to identify which columns hold:
  1. Road segment ID
  2. Distress class (e.g. pothole, alligator-crack)
  3. Severity level (low / moderate / high)
  4. Distress extent — either an existing rd_pct column or two columns to
     compute it from (extent ÷ road total × 100)

# =============================================================================
# CODE CONVENTIONS & OTHER INTERESTING TIDBITS
# =============================================================================
#
# LEADING UNDERSCORE ( _ )
#   Names beginning with _ are internal implementation details not intended to
#   be called or imported by outside code. The public-facing functions are:
#   - aggregate_measurements()
#   - assign_deduct_values() 
#   - calculate_segment_pci() 
#   - calculate_all_pci()   
#   Everything else is plumbing that supports those four.
#
# MODULE-LEVEL CONSTANTS
#   All-caps names (e.g. ALLIGATOR, SINGLE, CDV_TABLE) are lookup tables loaded
#   once at import time from the CSV files in this directory. Mixed-case _names
#   (e.g. _CDV_ARRAYS, _CLASS_TABLE) are derived structures built from those
#   tables — also computed once at startup, not recalculated per row.
#
# TYPE HINTS
#   Function signatures use Python type hints (e.g. float, pd.DataFrame,
#   dict[str, str]) as documentation. They are not enforced at runtime.
#
# RETURN STYLE
#   Functions return new objects rather than modifying inputs in place.
#   DataFrame functions receive a df, return a new df — the original is
#   unchanged (copies are made explicitly with .copy() where needed).
#
# ERROR HANDLING
#   Invalid or unmappable rows are dropped silently (via dropna or the
#   "__skip__" sentinel) rather than raising exceptions. A count of retained
#   rows is printed after mapping so you can spot unexpected data loss.
#
# =============================================================================

"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_TABLE_DIR = Path(__file__).parent

ALLIGATOR = pd.read_csv(_TABLE_DIR / "distress_alligator_area.csv")
SINGLE    = pd.read_csv(_TABLE_DIR / "distress_crack_length.csv")
PATCH     = pd.read_csv(_TABLE_DIR / "distress_patching_area.csv")
POTHOLE   = pd.read_csv(_TABLE_DIR / "distress_pothole_area.csv")
CDV_TABLE = pd.read_csv(_TABLE_DIR / "astm_cdv_correction_table.csv")

_CDV_ARRAYS: dict[int, tuple] = {
    q: (
        CDV_TABLE.loc[CDV_TABLE[f"q{q}_CDV"].notna(), "TDV"].to_numpy(float),
        CDV_TABLE.loc[CDV_TABLE[f"q{q}_CDV"].notna(), f"q{q}_CDV"].to_numpy(float),
    )
    for q in range(1, 8)
}

_CLASS_TABLE: dict[str, pd.DataFrame] = {
    "single-crack":    SINGLE,
    "crack-seal":      SINGLE,
    "alligator-crack": ALLIGATOR,
    "patch":           PATCH,
    "pothole":         POTHOLE,
}

_VALID_CLASSES    = list(_CLASS_TABLE.keys())
_VALID_SEVERITIES = {"low", "moderate", "high"}
_SEVERITY_RANK    = {"low": 0, "moderate": 1, "high": 2}
_GROUP_KEYS       = ["rd_cartid", "class_name", "severity"]
_MEASURE_COLS     = ["width_ft", "height_ft", "longest_ft", "area_sqft", "rd_pct"]

_PCI_RATINGS = [
    (85, "Good"), (70, "Satisfactory"), (55, "Fair"),
    (40, "Poor"), (25, "Very Poor"), (10, "Serious"), (0, "Failed"),
]


# ──────────────────────────────────────────────────────────────────────────────
'''ASTM D6433-24 sections 9.1 and 9.2'''

def aggregate_measurements(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Groups rows by (unique road ID, class_name, severity) and sums all measurement columns (width_ft, height_ft, etc.), 
    keeping the first value of road-dimension columns. Clips rd_pct at 100.
    '''
    measure_cols = [c for c in _MEASURE_COLS if c in df.columns]
    road_cols    = [c for c in ["rd_area", "rd_length", "rd_width"] if c in df.columns]
    num_df = df.copy()
    for c in measure_cols:
        num_df[c] = pd.to_numeric(num_df[c], errors="coerce")
    agg_spec = {c: (c, "sum") for c in measure_cols}
    agg_spec.update({c: (c, "first") for c in road_cols})
    grouped = (
        num_df
        .groupby(_GROUP_KEYS, sort=False)
        .agg(**agg_spec)
        .reset_index()
    )
    if "rd_pct" in grouped.columns:
        grouped["rd_pct"] = grouped["rd_pct"].clip(upper=100.0)
    return grouped.reindex(columns=[c for c in df.columns if c in grouped.columns], fill_value="")



'''ASTM D6433-24, section 9.3'''
def _interp_dv(table: pd.DataFrame, extent: float, severity: str) -> float:
    '''
    This function converts a distress extent percentage into a DV by linear interpolation, 
    using one of the four class-specific lookup tables (alligator, single-crack, patch, pothole). 
    Extrapolates linearly below the table's first row; clips at the table maximum.
    '''
    x = table["Extent"].values.astype(float)
    y = table[severity.capitalize()].values.astype(float)
    if extent <= 0:
        return 0.0
    if extent < x[0]:
        return round(extent / x[0] * y[0], 4)
    return round(float(np.interp(extent, x, y)), 4)


# ──────────────────────────────────────────────────────────────────────────────
''' ASTM D6433-24, section 9.5.2
Determine the allowable number of deducts, m, from Fig. 5, or using the following formula:
m = 1 + (9/98)(100 - HDV) 

where:
    m = allowable number of deducts including fractions (must be less than or equal to ten)
    HDV = highest individual deduct value.
'''
def calculate_m(hdv: float) -> float:
    '''
    This  function computes the allow number of deduct values from the highest deduct value (HDV)
    A high HDV gives a lower m, a low HDV allows up to m=10
    '''
    return min(1 + (9 / 98) * (100 - hdv), 10)


# ──────────────────────────────────────────────────────────────────────────────
'''ASTM D6433-24, section 9.5.3 and 9.5.4'''
def _apply_m_value(deduct_values: list) -> list:
    '''
    This trims and fractionally weights a sorted DV list based on the m-value of the highest DV.
    This ensures that only m deduct values (possibly with a fractional last one) enter the CDV calculation.
     '''
    if not deduct_values:
        return []
    sorted_dvs = sorted(deduct_values, reverse=True)
    m = max(1.0, calculate_m(sorted_dvs[0]))
    if len(sorted_dvs) <= m:
        return sorted_dvs
    floor_m = int(m)
    result = sorted_dvs[:floor_m]
    frac = m - floor_m
    if frac > 0 and floor_m < len(sorted_dvs):
        result = result + [frac * sorted_dvs[floor_m]]
    return result


# ──────────────────────────────────────────────────────────────────────────────
'''ASTM D6433-24, section 9.5.5'''
def get_cdv(tdv: float, q: int) -> float:
    ''' 
    This function looks up the Corrected Deduct Value (CDV) for a given Total Deduct Value (TDV) 
    and count of significant deducts (q) by interpolating into the pre-loaded CDV_TABLE
    '''
    if not 1 <= q <= 7:
        raise ValueError(f"q must be between 1 and 7, got {q}")
    xp, fp = _CDV_ARRAYS[q]
    return float(np.interp(min(tdv, xp[-1]), xp, fp))

def _iterative_cdv(deduct_values: list) -> float:
    '''This function runs the ASTM iterative CDV procedure: sort DVs, apply m-value, 
    then repeatedly compute CDV while replacing the smallest significant DV with 2.0 until only one remains. 
    Returns the maximum CDV observed across all iterations.'''
    dvs = sorted([float(d) for d in deduct_values if float(d) > 0], reverse=True)
    if not dvs:
        return 0.0
    dvs = _apply_m_value(dvs)
    max_cdv = 0.0
    while True:
        tdv = sum(dvs)
        q = sum(1 for d in dvs if d > 2.0)
        cdv = get_cdv(tdv, min(max(q, 1), 7))
        max_cdv = max(max_cdv, cdv)
        if q <= 1:
            break
        above = [(i, d) for i, d in enumerate(dvs) if d > 2.0]
        min_idx = min(above, key=lambda t: t[1])[0]
        dvs[min_idx] = 2.0
    return max_cdv


# ──────────────────────────────────────────────────────────────────────────────
'''ASTM D6433-24 section 9.6.2 '''

def _inv_interp_dv(table: pd.DataFrame, dv: float, severity: str) -> float:
    '''This is used in the next function ("combine_severity dvs") when there are multiple 
    severities of the same distress type. It returns an extent if given a DV'''
    col = severity.capitalize()
    x = table["Extent"].values.astype(float)
    y = table[col].values.astype(float)
    if dv <= 0:
        return 0.0
    if dv >= y[-1]:
        return float(x[-1])
    for i in range(len(y) - 1):
        if y[i] <= dv <= y[i + 1]:
            if y[i + 1] == y[i]:
                return float(x[i])
            return float(x[i] + (dv - y[i]) / (y[i + 1] - y[i]) * (x[i + 1] - x[i]))
    return 0.0


def _combine_severity_dvs(extents: dict) -> list:
    '''
    For each distress class that has multiple severity levels, this converts lower-severity 
    extents into equivalent higher-severity extents (via "_inv_interp_dv" avobe) and sums them, then 
    looks up the combined DV at the highest severity. Returns a list of DVs, one per distress class.
    '''
    by_class: dict = {}
    for (cls, sev), extent in extents.items():
        by_class.setdefault(cls, {})[sev] = extent
    dvs = []
    for cls, sev_extents in by_class.items():
        table = _CLASS_TABLE.get(cls)
        if table is None:
            continue
        if len(sev_extents) == 1:
            sev, extent = next(iter(sev_extents.items()))
            dvs.append(_interp_dv(table, extent, sev))
        else:
            highest = max(sev_extents, key=lambda s: _SEVERITY_RANK[s])
            combined = sev_extents.get(highest, 0.0)
            for sev, extent in sev_extents.items():
                if sev == highest:
                    continue
                dv_lower = _interp_dv(table, extent, sev)
                combined += _inv_interp_dv(table, dv_lower, highest)
            dvs.append(_interp_dv(table, combined, highest))
    return dvs


# ──────────────────────────────────────────────────────────────────────────────
''' ASTM D6433-24 sections 9.3 and 9.4'''

def assign_deduct_values(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Iterates the aggregated data frame from above row by row, looks up the appropriate class table, 
    and calls "_interp_dv" to compute a DV for each row. Appends a DV column and writes *_DV.csv
    '''
    dvs = []
    for _, row in df.iterrows():
        table = _CLASS_TABLE.get(str(row.get("class_name", "")).strip().lower())
        if table is None:
            dvs.append("")
            continue
        try:
            extent = float(row["rd_pct"])
        except (ValueError, TypeError):
            dvs.append("")
            continue
        severity = str(row.get("severity", "")).strip().lower()
        if severity not in _VALID_SEVERITIES:
            dvs.append("")
            continue
        dvs.append(_interp_dv(table, extent, severity))
    return df.copy().assign(DV=dvs)


# ──────────────────────────────────────────────────────────────────────────────
''' Standard PCI Rating Scale, ASTM D6433-24 Figure 1'''

def _pci_rating(pci: float) -> str:
    '''
    Assigns a qualitative value (good through failed) using a numeric PCI score
    '''
    for threshold, label in _PCI_RATINGS:
        if pci >= threshold:
            return label
    return "Failed"


# ──────────────────────────────────────────────────────────────────────────────
''' 
This combines the above functions step-wise to get a final PCI score when given
a list of road segments where each segment has one or more distresses present
'''

def calculate_segment_pci(segment_df: pd.DataFrame) -> dict:
    '''
    Computes PCI for a single road segment. Sums extents per (class_name, severity) group, 
    then runs both Method A (treat each severity separately) and Method B (combine severities) 
    through _iterative_cdv. Takes the higher CDV, computes PCI = 100 − max_cdv, and returns a 
    dictionary with pci, rating, max_cdv, cdv_a, cdv_b, and method used.
    '''
    valid = segment_df.copy()
    valid["rd_pct"] = pd.to_numeric(valid["rd_pct"], errors="coerce")
    valid = valid.dropna(subset=["rd_pct"])
    valid["class_name"] = valid["class_name"].str.strip().str.lower()
    valid["severity"]   = valid["severity"].str.strip().str.lower()
    valid = valid[
        valid["class_name"].isin(_CLASS_TABLE) &
        valid["severity"].isin(_VALID_SEVERITIES)
    ]
    if valid.empty:
        return {"pci": 100.0, "rating": "Good", "max_cdv": 0.0,
                "cdv_a": 0.0, "cdv_b": 0.0, "method": "none"}
    extents: dict = (
        valid.groupby(["class_name", "severity"])["rd_pct"]
        .sum()
        .to_dict()
    )
    dvs_a  = [_interp_dv(_CLASS_TABLE[cls], ext, sev) for (cls, sev), ext in extents.items()]
    dvs_b  = _combine_severity_dvs(extents)
    cdv_a  = _iterative_cdv(dvs_a)
    cdv_b  = _iterative_cdv(dvs_b)
    max_cdv = max(cdv_a, cdv_b)
    method  = "A" if cdv_a >= cdv_b else "B"
    pci = round(max(0.0, 100.0 - max_cdv), 1)
    return {
        "pci": pci, "rating": _pci_rating(pci),
        "max_cdv": round(max_cdv, 2),
        "cdv_a": round(cdv_a, 2), "cdv_b": round(cdv_b, 2),
        "method": method,
    }


def calculate_all_pci(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Iterates over all road segments (grouped by road's unique ID), calls calculate_segment_pci for each, 
    and returns a one-row-per-segment DataFrame.
    '''
    results = []
    for cart_id, group in df.groupby("rd_cartid"):
        r = calculate_segment_pci(group)
        r["rd_cartid"] = cart_id
        results.append(r)
    cols = ["rd_cartid", "pci", "rating", "max_cdv", "cdv_a", "cdv_b", "method"]
    return pd.DataFrame(results).reindex(columns=cols)


# ──────────────────────────────────────────────────────────────────────────────
# ── Interactive field mapping ─────────────────────────────────────────────────

def _pick_column(prompt: str, columns: list) -> str:
    """Print numbered column list and return the user's choice."""
    print(f"\n{prompt}")
    for i, col in enumerate(columns, 1):
        print(f"  {i:>3}. {col}")
    while True:
        raw = input("Enter number: ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(columns):
                return columns[idx]
        except ValueError:
            pass
        print("  Invalid choice, try again.")


def _map_values(col_name: str, unique_vals: list[str], valid_targets: list[str]) -> dict[str, str]:
    """Interactively map each unique value in a column to one of the valid targets."""
    print(f"\nMapping values in '{col_name}' to expected labels.")
    print(f"Expected labels: {', '.join(valid_targets)}")
    mapping: dict[str, str] = {}
    for val in unique_vals:
        # Auto-match if already valid (case-insensitive)
        lower = val.strip().lower()
        matched = next((t for t in valid_targets if t == lower), None)
        if matched:
            print(f"  '{val}'  →  '{matched}'  (auto-matched)")
            mapping[val] = matched
            continue
        # Prompt user
        print(f"\n  '{val}'  →  ?")
        for i, t in enumerate(valid_targets, 1):
            print(f"    {i}. {t}")
        print(f"    S. Skip (exclude rows with this value)")
        while True:
            raw = input("  Enter number or S: ").strip()
            if raw.upper() == "S":
                mapping[val] = "__skip__"
                break
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(valid_targets):
                    mapping[val] = valid_targets[idx]
                    break
            except ValueError:
                pass
            print("  Invalid choice, try again.")
    return mapping


def _compute_rd_pct(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Ask user to define rd_pct from two columns (extent / road total × 100)."""
    print("\nCompute rd_pct from two columns: (distress extent ÷ road total) × 100")
    extent_col = _pick_column("Which column holds the distress extent (area or length)?", columns)
    total_col  = _pick_column("Which column holds the total road area or length?", columns)
    extent = pd.to_numeric(df[extent_col], errors="coerce")
    total  = pd.to_numeric(df[total_col],  errors="coerce")
    pct = (extent / total * 100).clip(upper=100.0)
    n_bad = pct.isna().sum()
    if n_bad:
        print(f"  Warning: {n_bad} rows produced NaN rd_pct (non-numeric or zero denominator). "
              "Those rows will be excluded from PCI.")
    return pct


def run_interactive_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Guide the user through column mapping and return a normalized DataFrame."""
    columns = list(df.columns)
    print("\n" + "=" * 60)
    print("FIELD MAPPING")
    print("=" * 60)
    print(f"Detected {len(columns)} columns: {', '.join(columns)}")

    # ── 1. Segment ID ──────────────────────────────────────────────
    id_col = _pick_column(
        "Which column identifies each road segment (e.g. Cart_ID)?", columns
    )

    # ── 2. Distress class ─────────────────────────────────────────
    class_col = _pick_column(
        "Which column contains the distress class / type?", columns
    )
    unique_classes = df[class_col].dropna().unique().tolist()
    class_map = _map_values(class_col, unique_classes, _VALID_CLASSES)

    # ── 3. Severity ───────────────────────────────────────────────
    sev_col = _pick_column(
        "Which column contains the severity level?", columns
    )
    unique_sevs = df[sev_col].dropna().unique().tolist()
    sev_map = _map_values(sev_col, unique_sevs, sorted(_VALID_SEVERITIES))

    # ── 4. Extent (rd_pct) ────────────────────────────────────────
    remaining = [c for c in columns if c not in (id_col, class_col, sev_col)]
    print("\nHow is distress extent supplied?")
    print("  1. An existing rd_pct column (already a % of road area/length)")
    print("  2. Compute from two columns (extent ÷ road total × 100)")
    while True:
        raw = input("Enter 1 or 2: ").strip()
        if raw in ("1", "2"):
            extent_mode = raw
            break
        print("  Enter 1 or 2.")

    if extent_mode == "1":
        pct_col = _pick_column(
            "Which column is rd_pct (distress extent as % of road)?", remaining
        )
        rd_pct_series = pd.to_numeric(df[pct_col], errors="coerce")
    else:
        rd_pct_series = _compute_rd_pct(df, remaining)

    # ── 5. Build normalized DataFrame ────────────────────────────
    out = pd.DataFrame()
    out["rd_cartid"]  = df[id_col].astype(str)
    out["class_name"] = df[class_col].map(class_map)
    out["severity"]   = df[sev_col].map(sev_map)
    out["rd_pct"]     = rd_pct_series

    # Drop skipped / unmapped rows
    out = out[out["class_name"] != "__skip__"]
    out = out[out["severity"]   != "__skip__"]
    out = out.dropna(subset=["class_name", "severity", "rd_pct"])

    # Carry through any optional measurement columns if the user wants
    optional_candidates = [
        c for c in columns
        if c not in (id_col, class_col, sev_col)
        and c not in out.columns
    ]
    if optional_candidates:
        print("\nOptional: carry additional columns into the output?")
        print("  (These do not affect PCI but will appear in the DV and segPCI files.)")
        for i, c in enumerate(optional_candidates, 1):
            print(f"  {i:>3}. {c}")
        print(f"  {'A':>3}. All of the above")
        print(f"  {'N':>3}. None")
        raw = input("Enter numbers separated by commas, A, or N: ").strip().upper()
        if raw == "A":
            chosen = optional_candidates
        elif raw == "N":
            chosen = []
        else:
            chosen = []
            for tok in raw.split(","):
                tok = tok.strip()
                try:
                    idx = int(tok) - 1
                    if 0 <= idx < len(optional_candidates):
                        chosen.append(optional_candidates[idx])
                except ValueError:
                    pass
        for c in chosen:
            out[c] = df[c].values[: len(out)]  # align by position after filter

    print(f"\nMapping complete. {len(out)} rows retained after filtering.")
    return out.reset_index(drop=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    '''
    1. Parse --csv and --output CLI args.
    2. Read the raw CSV (all columns as strings).
    3. Call "run_interactive_mapping" → normalized DataFrame.
    4. "aggregate_measurements" → sum per segment/class/severity.
    5. "assign_deduct_values" → append DV column → write *_DV.csv.
    6. "calculate_all_pci" → per-segment PCI → join back to DV rows → write *_segPCI.csv and *_allPCI.csv.
    7. Print network average PCI and rating.
'''

    parser = argparse.ArgumentParser(
        description="Calculate ASTM D6433 PCI from any CSV with interactive field mapping."
    )
    parser.add_argument("csv", help="Path to input CSV")
    parser.add_argument(
        "--output", "-o",
        help="Output directory (default: same directory as input CSV)",
        default=None,
    )
    args = parser.parse_args()

    in_path = Path(args.csv)
    if not in_path.is_file():
        parser.error(f"File not found: {in_path}")

    out_dir = Path(args.output) if args.output else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / in_path.stem

    print(f"Reading {in_path.name} …")
    df_raw = pd.read_csv(in_path, dtype=str)
    if df_raw.empty or not len(df_raw.columns):
        sys.exit("ERROR: CSV appears empty or has no columns.")

    # Interactive mapping
    df = run_interactive_mapping(df_raw)

    if df.empty:
        sys.exit("No rows remain after mapping. Check that class and severity values were mapped correctly.")

    # 1. Aggregate per (rd_cartid, class_name, severity)
    df_agg = aggregate_measurements(df)

    # 2. Deduct values
    df_dv = assign_deduct_values(df_agg)
    out_dv = base.with_name(base.name + "_DV.csv")
    df_dv.to_csv(out_dv, index=False)
    print(f"\n  Written: {out_dv.name}")

    # 3. Segment PCI joined with DV rows
    df_pci = calculate_all_pci(df_dv)
    df_segpci = df_dv.merge(df_pci, on="rd_cartid", how="left")
    df_segpci = df_segpci.replace("", pd.NA).dropna(axis=1, how="all").fillna("")
    out_seg = base.with_name(base.name + "_segPCI.csv")
    df_segpci.to_csv(out_seg, index=False)
    print(f"  Written: {out_seg.name}")

    # 4. One-row-per-segment summary
    out_all = base.with_name(base.name + "_allPCI.csv")
    df_pci.to_csv(out_all, index=False)
    print(f"  Written: {out_all.name}")

    print(f"\nDone. {len(df_pci)} road segments processed.")
    avg_pci = df_pci["pci"].mean()
    print(f"Network average PCI: {avg_pci:.1f}  ({_pci_rating(avg_pci)})")


if __name__ == "__main__":
    '''if given a csv '''
    main()
