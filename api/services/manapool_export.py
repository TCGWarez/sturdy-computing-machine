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

def _map_finish_for_manapool(raw_finish: str) -> str:
    """
    Map internal finish to Manapool finish code (NF/FO/EF).
    Uses simple heuristics - no hardcoded mapping table needed.
    """
    key = (raw_finish or "").lower().strip()

    if "etched" in key:
        return "EF"
    if "foil" in key or "surge" in key:
        return "FO"
    return "NF"


def _match_sku_for_finish(skus: List[Dict], local_finish: str) -> Optional[Dict]:
    """
    Match the correct SKU from MTGSold results based on local card finish.

    MTGSold returns all SKUs for a scryfall_id. We match by finish category:
    - 'etched' -> SKU where finish contains 'ETCHED'
    - 'foil' -> SKU where finish ends with 'FOIL' (but not 'NON FOIL')
    - 'nonfoil' -> SKU where finish contains 'NON FOIL'
    """
    local_lower = (local_finish or "").lower().strip()

    for sku in skus:
        mtg_finish = (sku.get("finish") or "").upper()

        if "etched" in local_lower and "ETCHED" in mtg_finish:
            return sku
        elif local_lower == "foil" and mtg_finish.endswith("FOIL") and "NON FOIL" not in mtg_finish:
            return sku
        elif local_lower == "nonfoil" and "NON FOIL" in mtg_finish:
            return sku

    return None


def _mtgsold_headers() -> Dict[str, str]:
    """Build headers for MTGSold API with optional bearer auth."""
    headers: Dict[str, str] = {}
    if MTGSOLD_API_TOKEN is not None:
        headers["Authorization"] = f"Bearer {MTGSOLD_API_TOKEN}"
    return headers


def _batch_fetch_mtgsold_skus(scryfall_ids: List[str]) -> Dict[str, List[Dict]]:
    """
    Fetch ALL SKUs for given scryfall_ids (no finish filter).

    Returns: scryfall_id -> list of SKU records

    This approach is more robust than filtering by exact finish string because:
    - MTGSold finish strings include variant info (BORDERLESS SHOWCASE, etc.)
    - We don't need to construct exact finish strings
    - We can match by finish category (foil/nonfoil/etched) from the results
    """
    results: Dict[str, List[Dict]] = {}

    if MTGSOLD_API_TOKEN is None or not scryfall_ids:
        return results

    # Deduplicate scryfall_ids
    unique_ids = list(set(scryfall_ids))

    # PostgREST in list - query without finish filter
    in_list = ",".join(unique_ids)
    params = {
        "select": "scryfall_id,mp_product_id,market_price,condition_short,finish,language",
        "scryfall_id": f"in.({in_list})",
        "language": f"eq.{MANAPOOL_LANGUAGE_OUTPUT}",
    }

    try:
        response = requests.get(
            f"{MTGSOLD_API_BASE}/api/skus",
            params=params,
            headers=_mtgsold_headers(),
            timeout=20,
        )
        if not response.ok:
            return results

        payload = response.json()
        records = payload.get("data") if isinstance(payload, dict) else payload
        if not records:
            return results

        # Group by scryfall_id (each card may have multiple SKUs for different finishes)
        for record in records:
            sid = record.get("scryfall_id")
            if sid:
                results.setdefault(sid, []).append(record)
    except Exception:
        pass

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

    # Collect all scryfall_ids for batch MTGSold lookup
    scryfall_ids = [
        meta.get("scryfall_id")
        for meta in consolidated.values()
        if meta.get("scryfall_id")
    ]

    # Fetch all SKUs for these cards (no finish filter - we'll match by category)
    mtgsold_lookup = _batch_fetch_mtgsold_skus(scryfall_ids)

    manapool_rows = []
    for key, meta in consolidated.items():
        set_code, collector_number, manapool_finish, language_out, condition_out, name = key
        scryfall_id = meta.get("scryfall_id")
        finish_raw = meta.get("finish_raw")

        mp_product_id = market_price = api_condition = api_language = None
        if scryfall_id and scryfall_id in mtgsold_lookup:
            # Match the correct SKU based on our finish category (foil/nonfoil/etched)
            skus = mtgsold_lookup[scryfall_id]
            rec = _match_sku_for_finish(skus, finish_raw)
            if rec:
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

