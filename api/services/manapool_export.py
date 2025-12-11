"""
Manapool CSV export service.
Encapsulates MTGSold enrichment and CSV generation to keep routes lean.
"""

import csv
import io
import os
from typing import Dict, List, Optional, Tuple

import requests

from api.database import BatchResult
from src.database.schema import Card as DBCard

# MTGSold configuration and constants
MTGSOLD_API_BASE = os.getenv("MTGSOLD_API_BASE", "https://api.mtgsold.com")
MTGSOLD_API_TOKEN = os.getenv(
    "MTGSOLD_API_TOKEN",
)

# Manapool output defaults
MANAPOOL_PRODUCT_TYPE = "mtg_single"
MANAPOOL_LANGUAGE_OUTPUT = "ENGLISH"
MANAPOOL_CONDITION_OUTPUT = "NM"

# Finish mappings (keyed by lowercase finish string for lookup)
# Each entry maps to a tuple: (manapool_code, mtgsold_finish_string)
MTGSOLD_FINISH_MAP: Dict[str, Tuple[str, str]] = {
    # Base finishes
    "nonfoil": ("NF", "NON FOIL"),
    "non foil": ("NF", "NON FOIL"),
    "foil": ("FO", "FOIL"),
    "etched": ("EF", "FOIL ETCHED"),
    "etched foil": ("EF", "FOIL ETCHED"),
    "foil etched": ("EF", "FOIL ETCHED"),
    # variant types
    "borderless foil": ("FO", "BORDERLESS FOIL"),
    "borderless foil etched": ("EF", "BORDERLESS FOIL ETCHED"),
    "borderless non foil": ("NF", "BORDERLESS NON FOIL"),
    "borderless showcase foil": ("FO", "BORDERLESS SHOWCASE FOIL"),
    "borderless showcase foil etched": ("EF", "BORDERLESS SHOWCASE FOIL ETCHED"),
    "borderless showcase non foil": ("NF", "BORDERLESS SHOWCASE NON FOIL"),
    "borderless showcase surge foil": ("FO", "BORDERLESS SHOWCASE SURGE FOIL"),
    "borderless surge foil": ("FO", "BORDERLESS SURGE FOIL"),
    "collector edition": ("NF", "COLLECTOR EDITION"),
    "galaxy borderless foil": ("FO", "GALAXY BORDERLESS FOIL"),
    "galaxy borderless non foil": ("NF", "GALAXY BORDERLESS NON FOIL"),
    "galaxy borderless showcase foil": ("FO", "GALAXY BORDERLESS SHOWCASE FOIL"),
    "galaxy foil": ("FO", "GALAXY FOIL"),
    "galaxy showcase foil": ("FO", "GALAXY SHOWCASE FOIL"),
    "halo borderless foil": ("FO", "HALO BORDERLESS FOIL"),
    "halo borderless showcase foil": ("FO", "HALO BORDERLESS SHOWCASE FOIL"),
    "halo showcase foil": ("FO", "HALO SHOWCASE FOIL"),
    "non foil": ("NF", "NON FOIL"),
    "oil slick raised borderless foil": ("FO", "OIL SLICK RAISED BORDERLESS FOIL"),
    "rainbow borderless foil": ("FO", "RAINBOW BORDERLESS FOIL"),
    "rainbow borderless non foil": ("NF", "RAINBOW BORDERLESS NON FOIL"),
    "rainbow borderless showcase foil": ("FO", "RAINBOW BORDERLESS SHOWCASE FOIL"),
    "rainbow borderless showcase non foil": ("NF", "RAINBOW BORDERLESS SHOWCASE NON FOIL"),
    "rainbow foil": ("FO", "RAINBOW FOIL"),
    "rainbow non foil": ("NF", "RAINBOW NON FOIL"),
    "rainbow showcase foil": ("FO", "RAINBOW SHOWCASE FOIL"),
    "raised borderless foil": ("FO", "RAISED BORDERLESS FOIL"),
    "raised borderless showcase foil": ("FO", "RAISED BORDERLESS SHOWCASE FOIL"),
    "raised showcase foil": ("FO", "RAISED SHOWCASE FOIL"),
    "showcase foil": ("FO", "SHOWCASE FOIL"),
    "showcase nonfoil": ("NF", "SHOWCASE NON FOIL"),
    "showcase non foil": ("NF", "SHOWCASE NON FOIL"),
    "showcase surge foil": ("FO", "SHOWCASE SURGE FOIL"),
    "surge foil": ("FO", "SURGE FOIL"),
}

def _normalize_finish(raw_finish: str) -> str:
    """
    Normalize finish string for lookup (handle nonfoil/no-space variants).
    """
    normalized = (raw_finish or "").lower().strip()
    if normalized == "nonfoil":
        normalized = "non foil"
    return normalized


def _finish_candidate_keys(raw_finish: str, variant_type: Optional[str]) -> List[str]:
    """
    Build candidate lookup keys ordered by specificity.
    Includes variant-type combinations (showcase/borderless) when present.
    """
    key = _normalize_finish(raw_finish)
    variant = (variant_type or "").lower().strip()

    candidates: List[str] = []
    # Variant-specific permutations (most specific first)
    if variant:
        if "borderless" in variant and "showcase" in variant:
            candidates.append(f"borderless showcase {key}")
        if "borderless" in variant:
            candidates.append(f"borderless {key}")
        if "showcase" in variant:
            candidates.append(f"showcase {key}")
    # Base key
    candidates.append(key)
    # Deduplicate while preserving order
    seen = set()
    ordered: List[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _map_finish_for_manapool(raw_finish: str) -> str:
    """
    Map internal finish to Manapool finish code; default to NF if unknown.
    """
    key = (raw_finish or "").lower().strip()
    if key in MTGSOLD_FINISH_MAP:
        return MTGSOLD_FINISH_MAP[key][0]

    # Fallback heuristic: etched > foil/surge > nonfoil
    if "etched" in key:
        return "EF"
    if "foil" in key or "surge" in key:
        return "FO"
    return "NF"


def _map_finish_for_mtgsold(raw_finish: str, variant_type: Optional[str]) -> Optional[str]:
    """
    Map internal finish to MTGSold finish string; return None when unmapped.
    """
    key = (raw_finish or "").lower().strip()
    for cand in _finish_candidate_keys(raw_finish, variant_type):
        if cand in MTGSOLD_FINISH_MAP:
            return MTGSOLD_FINISH_MAP[cand][1]

    # Fallback heuristic: etched > foil/surge > nonfoil
    if "etched" in key:
        return "FOIL ETCHED"
    if "foil" in key or "surge" in key:
        return "FOIL"
    return None


def _mtgsold_headers() -> Dict[str, str]:
    """Build headers for MTGSold API with optional bearer auth."""
    headers: Dict[str, str] = {}
    if MTGSOLD_API_TOKEN is not None:
        headers["Authorization"] = f"Bearer {MTGSOLD_API_TOKEN}"
    return headers


def _batch_fetch_mtgsold_skus(grouped: Dict[str, List[str]]) -> Dict[str, Dict[str, object]]:
    """
    Batch fetch MTGSold SKUs using PostgREST in-filter.
    grouped: mtgsold_finish -> list of scryfall_ids.
    Returns mapping scryfall_id -> MTGSold fields.
    """
    results: Dict[str, Dict[str, object]] = {}

    if MTGSOLD_API_TOKEN is None:
        return results

    for mtgsold_finish, scryfall_ids in grouped.items():
        if not scryfall_ids:
            continue

        # PostgREST in list
        in_list = ",".join(scryfall_ids)
        params = {
            "select": "scryfall_id,mp_product_id,market_price,condition_short,finish,language",
            "scryfall_id": f"in.({in_list})",
            "finish": f"eq.{mtgsold_finish}",
            "language": f"eq.{MANAPOOL_LANGUAGE_OUTPUT}",
            "limit": len(scryfall_ids),
        }

        try:
            response = requests.get(
                f"{MTGSOLD_API_BASE}/api/skus",
                params=params,
                headers=_mtgsold_headers(),
                timeout=20,
            )
            if not response.ok:
                continue

            payload = response.json()
            records = payload.get("data") if isinstance(payload, dict) else payload
            if not records:
                continue

            for record in records:
                sid = record.get("scryfall_id")
                if sid:
                    results[sid] = record
        except Exception:
            continue

    return results


def build_manapool_csv(resolved_rows: List[Tuple[BatchResult, DBCard]]) -> str:
    """
    Build Manapool CSV string from resolved batch rows.
    """
    consolidated: Dict[
        Tuple[str, str, str, str, str, str],
        Dict[str, object],
    ] = {}

    for _, card in resolved_rows:
        manapool_finish = _map_finish_for_manapool(card.finish)
        key = (
            card.set_code,
            card.collector_number,
            manapool_finish,
            MANAPOOL_LANGUAGE_OUTPUT,
            MANAPOOL_CONDITION_OUTPUT,
            card.name,
        )
        if key not in consolidated:
            consolidated[key] = {
                "quantity": 0,
                "finish_raw": card.finish,
                "scryfall_id": getattr(card, "scryfall_id", None),
                "variant_type": getattr(card, "variant_type", None),
            }
        consolidated[key]["quantity"] += 1

    # Batch MTGSold lookups grouped by mapped MTGSold finish to avoid per-row calls
    grouped_by_mtgsold_finish: Dict[str, List[str]] = {}
    for meta in consolidated.values():
        finish_raw = meta.get("finish_raw")
        variant_type = meta.get("variant_type")
        scryfall_id = meta.get("scryfall_id")
        if not scryfall_id:
            continue
        mtgsold_finish = _map_finish_for_mtgsold(finish_raw, variant_type)
        if not mtgsold_finish:
            continue
        grouped_by_mtgsold_finish.setdefault(mtgsold_finish, []).append(scryfall_id)

    mtgsold_lookup = _batch_fetch_mtgsold_skus(grouped_by_mtgsold_finish)

    manapool_rows = []
    for key, meta in consolidated.items():
        set_code, collector_number, manapool_finish, language_out, condition_out, name = key
        scryfall_id = meta.get("scryfall_id")
        finish_raw = meta.get("finish_raw")
        variant_type = meta.get("variant_type")

        mp_product_id = market_price = api_condition = api_language = None
        if scryfall_id and scryfall_id in mtgsold_lookup:
            rec = mtgsold_lookup[scryfall_id]
            mp_product_id = rec.get("mp_product_id")
            market_price = rec.get("market_price")
            api_condition = rec.get("condition_short")
            api_language = rec.get("language")

        condition_value = api_condition or condition_out
        language_value = api_language or language_out

        manapool_rows.append([
            MANAPOOL_PRODUCT_TYPE,
            mp_product_id or "",
            name,
            set_code,
            collector_number,
            language_value,
            manapool_finish,
            condition_value,
            market_price if market_price is not None else "",
            meta["quantity"],
        ])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "product_type",
        "product_id",
        "name",
        "set",
        "number",
        "language",
        "finish",
        "condition",
        "price",
        "quantity",
    ])
    for row in manapool_rows:
        writer.writerow(row)

    output.seek(0)
    return output.getvalue()

