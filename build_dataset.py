"""
Rebuilds the modelling table from the NOAA Oil Library JSON records.

Each record is a crude-oil assay. Properties are reported at several reference
temperatures and, critically, in several different units within the same field:
density appears as g/mL, g/cm^3 and kg/m^3; dynamic viscosity as mPa.s, cP and
kg/(m s); pour and flash points in Celsius and Kelvin; SARA fractions as both
percentages and fractions. Every value is converted to a single unit here --
without that the columns are meaningless mixtures.

Output: dataset.csv
"""

import json
import os
from collections import Counter

import pandas as pd

FOLDER = "oil"
TARGET_TEMP = 15.0  # degrees C

# --- unit conversion tables: factor/offset into the canonical unit -----------

DENSITY_TO_G_PER_ML = {          # canonical: g/mL
    "g/mL": 1.0,
    "g/cm^3": 1.0,
    "g/cm³": 1.0,
    "kg/m^3": 1e-3,
}

VISCOSITY_TO_MPA_S = {           # canonical: mPa*s  (1 cP == 1 mPa*s)
    "mPas": 1.0,
    "mPa.s": 1.0,
    "mPa s": 1.0,
    "cP": 1.0,
    "Pa.s": 1e3,
    "kg/(m s)": 1e3,             # Pa*s expressed in base SI units
}

FRACTION_TO_UNIT_FRACTION = {    # canonical: fraction in [0, 1]
    "fraction": 1.0,
    "%": 1e-2,
}

skipped_units = Counter()


def convert(value, unit, table, field):
    """Scale `value` into the canonical unit, or drop it if the unit is unknown."""
    if value is None:
        return None
    if unit not in table:
        skipped_units[f"{field}:{unit}"] += 1
        return None
    return value * table[unit]


def to_celsius(value, unit, field):
    if value is None:
        return None
    if unit == "C":
        return value
    if unit == "K":
        return value - 273.15
    skipped_units[f"{field}:{unit}"] += 1
    return None


def closest_by_temp(items):
    """Return the record measured nearest to TARGET_TEMP, or None."""
    best, best_diff = None, float("inf")
    for item in items:
        temp = item.get("ref_temp", {}).get("value")
        temp_unit = item.get("ref_temp", {}).get("unit")
        if temp is None:
            continue
        if temp_unit == "K":
            temp = temp - 273.15
        diff = abs(temp - TARGET_TEMP)
        if diff < best_diff:
            best, best_diff = item, diff
    return best


def get_density(items):
    item = closest_by_temp(items)
    if not item:
        return None
    d = item.get("density", {})
    return convert(d.get("value"), d.get("unit"), DENSITY_TO_G_PER_ML, "density")


def get_viscosity(items):
    item = closest_by_temp(items)
    if not item:
        return None
    v = item.get("viscosity", {})
    return convert(v.get("value"), v.get("unit"), VISCOSITY_TO_MPA_S, "viscosity")


def get_temperature(phys, key):
    m = (phys.get(key) or {}).get("measurement") or {}
    return to_celsius(m.get("value"), m.get("unit"), key)


def get_sara(sample):
    sara = sample.get("SARA", {})
    out = {}
    for name in ("saturates", "aromatics", "resins", "asphaltenes"):
        entry = sara.get(name) or {}
        out[name] = convert(entry.get("value"), entry.get("unit"),
                            FRACTION_TO_UNIT_FRACTION, f"sara_{name}")
    return out


def get_sulfur(sample):
    for item in sample.get("bulk_composition", []):
        if item.get("name") == "Sulfur Content":
            m = item.get("measurement", {})
            return convert(m.get("value"), m.get("unit"),
                           FRACTION_TO_UNIT_FRACTION, "sulfur")
    return None


def classify_oil(api):
    """Standard API-gravity bands used by the industry."""
    if api > 31:
        return "light"
    if api > 22:
        return "medium"
    return "heavy"


def main():
    rows, skipped = [], 0

    for root, _, files in os.walk(FOLDER):
        for file in files:
            if not file.endswith(".json"):
                continue
            with open(os.path.join(root, file)) as f:
                try:
                    oil = json.load(f)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

            api = (oil.get("metadata") or {}).get("API")
            sub_samples = oil.get("sub_samples") or []
            if api is None or not sub_samples:
                skipped += 1
                continue

            sample = sub_samples[0]
            phys = sample.get("physical_properties", {}) or {}

            density = get_density(phys.get("densities", []))
            viscosity = get_viscosity(phys.get("dynamic_viscosities", []))
            if density is None or viscosity is None:
                skipped += 1
                continue

            rows.append({
                "API": api,
                "density": density,
                "viscosity": viscosity,
                "pour_point": get_temperature(phys, "pour_point"),
                "flash_point": get_temperature(phys, "flash_point"),
                **get_sara(sample),
                "sulfur": get_sulfur(sample),
                "class": classify_oil(api),
            })

    df = pd.DataFrame(rows)
    df.to_csv("dataset.csv", index=False)

    print(f"assays parsed : {len(df)}")
    print(f"assays skipped: {skipped}  (missing API, density or viscosity)")
    if skipped_units:
        print(f"values dropped on unrecognised units: {dict(skipped_units)}")
    print("\nclass balance:")
    print(df["class"].value_counts().to_string())
    print("\ncoverage (share of rows present):")
    print((1 - df.isna().mean()).round(3).to_string())
    print("\ncanonical-unit sanity check:")
    print(df[["density", "viscosity", "pour_point", "flash_point"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
