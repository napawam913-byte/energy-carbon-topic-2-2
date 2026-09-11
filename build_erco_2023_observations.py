"""Build hourly observations for the fixed ERCO 2023 pilot, not forecasting features.

Requires pandas and numpy. Keeps raw inputs unchanged. No network or training.
Run: python build_erco_2023_observations.py --root /workspace/energy-carbon
"""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path("/workspace/energy-carbon"))
args = parser.parse_args()
base = args.root.resolve()
root = base / "data/raw/oge/v0.8.0/2023"
outdir = base / "data/processed/erco_2023_v1"
if outdir.exists():
    raise FileExistsError(f"Refusing to overwrite existing output directory: {outdir}")

expected = {
    "power_sector_data": "42f8086ef1af21e34afb1775e7e9413ada1e5296149fab545693989dc9318a64",
    "carbon_accounting": "2b9d8ef20cc5395179a848f1913003ec04e850c265b80d739735b3437ee2c174",
}
source_hashes = {}
for name, digest in expected.items():
    path = root / name / "ERCO.csv"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise ValueError(f"Unexpected input bytes for fixed v0.8.0 pilot: {path}")
    source_hashes[str(path.relative_to(base))] = actual

g = pd.read_csv(root / "power_sector_data/ERCO.csv", low_memory=False)
carbon = pd.read_csv(root / "carbon_accounting/ERCO.csv", low_memory=False)
assert g["fuel_category"].notna().all(), "Null fuel category"

q = "net_generation_mwh"
c = "co2_mass_kg_for_electricity"
g["timestamp_utc"] = pd.to_datetime(g["datetime_utc"], utc=True, errors="raise")
assert not g.duplicated(["timestamp_utc", "fuel_category"]).any()
parts = g[g["fuel_category"] != "total"].copy()
fuels = ["biomass", "coal", "hydro", "natural_gas", "nuclear",
         "other", "petroleum", "solar", "wind"]
assert set(parts["fuel_category"]) == set(fuels)

Q = parts.pivot(index="timestamp_utc", columns="fuel_category", values=q)[fuels].sort_index()
C = parts.pivot(index="timestamp_utc", columns="fuel_category", values=c)[fuels].sort_index()
assert np.isfinite(Q.to_numpy()).all() and np.isfinite(C.to_numpy()).all()
assert Q.sum(axis=1).gt(0).all()

# Individual factors are undefined when generation is zero or negative.
F = C.div(Q.where(Q > 0))
data = pd.concat([
    Q.add_prefix("generation_").add_suffix("_mwh"),
    C.add_prefix("co2_").add_suffix("_kg"),
    F.add_prefix("factor_").add_suffix("_kg_per_mwh"),
], axis=1)
data["generation_total_mwh"] = Q.sum(axis=1)
data["co2_total_kg"] = C.sum(axis=1)
data["factor_generated_kg_per_mwh"] = data["co2_total_kg"] / data["generation_total_mwh"]
data["flag_negative_generation"] = Q.lt(0).any(axis=1).astype("int8")
ref = g[g["fuel_category"] == "total"].set_index("timestamp_utc")
data["reference_generated_factor_kg_per_mwh"] = ref["generated_co2_rate_kg_per_mwh_for_electricity"]

carbon["timestamp_utc"] = pd.to_datetime(carbon["datetime_utc"], utc=True, errors="raise")
consumed = carbon.set_index("timestamp_utc")[[
    "consumed_co2_rate_kg_per_mwh_for_electricity",
    "consumed_co2_rate_kg_per_mwh_for_electricity_adjusted",
]].rename(columns={
    "consumed_co2_rate_kg_per_mwh_for_electricity": "reference_consumed_factor_kg_per_mwh",
    "consumed_co2_rate_kg_per_mwh_for_electricity_adjusted": "reference_consumed_adjusted_factor_kg_per_mwh",
})
assert consumed.index.is_unique
assert consumed.index.sort_values().equals(data.index)
assert np.isfinite(consumed.to_numpy()).all()
data = data.join(consumed, how="left", validate="one_to_one")

grid = pd.date_range("2023-01-01 06:00:00", periods=8760, freq="h", tz="UTC")
assert data.index.equals(grid), "OGE hourly coverage mismatch"

# EIA end-of-hour timestamps must move back one hour to OGE start-of-hour.
eia_columns = {
    "Demand (MW)": "demand_mw",
    "Net Generation (MW) from Wind": "reference_eia_wind_mw",
    "Net Generation (MW) from Solar": "reference_eia_solar_mw",
    "Net Generation (MW) from Coal": "reference_eia_coal_mw",
    "Net Generation (MW) from Natural Gas": "reference_eia_natural_gas_mw",
}
time_col = "UTC Time at End of Hour"
eia_chunks = []
for filename in ["EIA930_BALANCE_2023_Jan_Jun.csv", "EIA930_BALANCE_2023_Jul_Dec.csv"]:
    path = base / "data/raw/eia" / filename
    source_hashes[str(path.relative_to(base))] = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_csv(
        path,
        usecols=["Balancing Authority", time_col] + list(eia_columns),
        low_memory=False,
    )
    frame = frame.loc[frame["Balancing Authority"].eq("ERCO")].copy()
    assert not frame.empty, f"No ERCO rows: {filename}"
    print(f"{filename}: {len(frame)} ERCO rows")
    eia_chunks.append(frame)

eia = pd.concat(eia_chunks, ignore_index=True).rename(columns=eia_columns)
eia["eia_hour_end_utc"] = pd.to_datetime(
    eia[time_col], format="%m/%d/%Y %I:%M:%S %p", utc=True, errors="raise"
)
eia["timestamp_utc"] = eia["eia_hour_end_utc"] - pd.Timedelta(hours=1)
assert eia["timestamp_utc"].notna().all(), "Null EIA timestamp"
assert not eia["timestamp_utc"].duplicated().any(), "Duplicate EIA hour"
eia = eia.set_index("timestamp_utc").sort_index()
assert eia.index.equals(grid), "EIA hourly coverage mismatch"
assert np.isfinite(eia[list(eia_columns.values())].to_numpy()).all()
assert eia["demand_mw"].gt(0).all(), "Non-positive demand"

data = data.join(
    eia[list(eia_columns.values()) + ["eia_hour_end_utc"]],
    how="left",
    validate="one_to_one",
)
assert len(data) == 8760 and data.index.is_unique
assert np.isfinite(data["factor_generated_kg_per_mwh"].to_numpy()).all()
# Source-specific factor NaNs at zero/negative generation are intentional.
assert data.drop(columns=list(F.add_prefix("factor_").add_suffix("_kg_per_mwh").columns)).notna().all().all()

negative = parts.loc[parts[q] < 0].sort_values("timestamp_utc")
mass_difference = (data["co2_total_kg"] - ref[c]).abs().max()
energy_difference = (data["generation_total_mwh"] - ref[q]).abs().max()
metadata = {
    "status": "OBSERVATIONS_BUILT_NOT_FORECAST_VALIDATED",
    "region": "ERCO",
    "dataset_year": 2023,
    "oge_version": "v0.8.0",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "hour_start_utc_min": str(data.index.min()),
    "hour_start_utc_max": str(data.index.max()),
    "hourly_rows": len(data),
    "csv_columns_including_timestamp": len(data.columns) + 1,
    "source_sha256": source_hashes,
    "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    "negative_generation_records": len(negative),
    "negative_generation_min_mwh": None if negative.empty else float(negative[q].min()),
    "negative_generation_max_mwh": None if negative.empty else float(negative[q].max()),
    "undefined_source_factor_counts": {k: int(v) for k, v in F.isna().sum().items()},
    "max_component_sum_difference_from_total": {
        "net_generation_mwh": float(energy_difference),
        "co2_mass_kg": float(mass_difference),
    },
    "definitions": {
        "generation_*_mwh": "OGE net electricity generation, by fuel; negatives retained.",
        "co2_*_kg": "OGE CO2 mass attributed to electricity; no adjusted suffix; not CO2e.",
        "factor_<fuel>_kg_per_mwh": "Per-fuel electricity CO2 mass / net generation, defined only where generation > 0.",
        "factor_generated_kg_per_mwh": "Sum of nine OGE fuel CO2 masses / sum of nine OGE net generation values; total rows excluded.",
        "demand_mw": "EIA original Demand (MW), not adjusted or imputed demand.",
        "reference_eia_*_mw": "EIA original source generation, retained separately, not used as the OGE factor denominator.",
        "reference_*factor*": "Published OGE factors retained for comparison, not independent validation.",
        "flag_negative_generation": "At least one OGE fuel has negative net generation in this hour.",
    },
    "notes": [
        "Regional electric-generation benchmark; not measured industrial-park data or a multi-carrier heat/gas inventory.",
        "These are contemporaneous historical observations, NOT ready-made forecasting inputs.",
        "Do not expose future actual generation, emissions, factors or quality flags as forecast features.",
        "OGE is retrospective; real-time availability and publication delays have not been validated.",
        "Generation-based and consumption-based factors have different boundaries and must not be conflated.",
        "Negative generation is retained and flagged; its record-specific cause has not been established.",
        "Source-specific factor NaNs at non-positive generation are intentional; do not blanket-fill zeros or drop these hours.",
        "No train/validation/test split, imputation, normalization, model fitting or performance evaluation was performed.",
    ],
}
outdir.mkdir(parents=True, exist_ok=False)
data.reset_index().to_csv(outdir / "observations.csv", index=False, mode="x")
negative.to_csv(outdir / "negative_generation_audit.csv", index=False, mode="x")
with (outdir / "metadata.json").open("x", encoding="utf-8") as handle:
    json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)

# Re-read written tables to verify row count and numerical round trip.
saved = pd.read_csv(outdir / "observations.csv")
assert len(saved) == 8760 and len(saved.columns) == len(data.columns) + 1
saved_time = pd.DatetimeIndex(pd.to_datetime(saved["timestamp_utc"], utc=True))
assert saved_time.equals(grid)
np.testing.assert_allclose(
    saved[data.select_dtypes(include="number").columns].to_numpy(),
    data.select_dtypes(include="number").to_numpy(),
    rtol=1e-12, atol=1e-10, equal_nan=True,
)
print("\nObservation rows:", len(saved), "columns:", len(saved.columns))
print("Negative generation rows (retained):", len(negative))
print("Undefined per-source factors:", F.isna().sum().to_dict())
print("\nFirst three rows:")
print(data[["demand_mw", "generation_wind_mwh", "generation_solar_mwh",
            "factor_generated_kg_per_mwh"]].head(3).to_string())
print("\nSaved to:", outdir)
print("PASS: observation table built and re-read; no training performed.")

