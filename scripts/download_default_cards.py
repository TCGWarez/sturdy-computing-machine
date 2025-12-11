#!/usr/bin/env python3
"""
scripts/download_default_cards.py: Download Scryfall reference images
Stream-parse default_cards.json and download English card images

Also saves JSON metadata sidecar files for each image containing the full
Scryfall card data (including scryfall_id UUID).
"""

import os
import sys
import json
import asyncio
import aiohttp
import aiofiles
import ijson
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from tqdm.asyncio import tqdm_asyncio
import logging
import dotenv

dotenv.load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SCRYFALL_BULK_URL = "https://api.scryfall.com/bulk-data"
# DEFAULT_CARDS_URL = "https://data.scryfall.io/default-cards/default-cards-{}.json"
DATA_DIR = Path(os.getenv("SCRYFALL_IMAGES_DIR", "data/scryfall"))
CONCURRENT_DOWNLOADS = 10


def decimal_default(obj):
    """JSON serializer for Decimal objects from ijson"""
    from decimal import Decimal
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


async def save_json_metadata(filepath: Path, card_json: Dict[str, Any]) -> bool:
    """Save card JSON metadata as sidecar file"""
    json_path = filepath.with_suffix('.json')
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(json_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(card_json, indent=2, default=decimal_default))
        return True
    except Exception as e:
        logger.error(f"Error saving JSON {json_path}: {e}")
        return False

CACHE_DIR = Path(os.getenv("CACHE_DIR", "data/cache"))
async def fetch_bulk_data_info() -> str:
    """Fetch (with caching) the latest default_cards download URL from Scryfall."""
   
    CACHE_FILE = CACHE_DIR 
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                cached = json.load(f)

            if "default_cards_url" in cached:
                return cached["default_cards_url"]

        except Exception:
            CACHE_FILE.unlink(missing_ok=True)

    
    async with aiohttp.ClientSession() as session:
        async with session.get(SCRYFALL_BULK_URL) as response:
            data = await response.json()

            for item in data["data"]:
                if item["type"] == "default_cards":
                    url = item["download_uri"]

                   
                    with open(CACHE_FILE, "w") as f:
                        json.dump({"default_cards_url": url}, f)

                    return url

    raise ValueError("Could not find default_cards bulk data URL")


async def download_image_and_json(
    session: aiohttp.ClientSession,
    url: str,
    filepath: Path,
    card_json: Dict[str, Any],
    semaphore: asyncio.Semaphore
) -> Tuple[bool, bool]:
    """
    Download image and save JSON metadata sidecar.

    Returns:
        Tuple of (image_success, json_success)
        - Skips image download if file exists
        - Creates JSON if missing even if image exists
    """
    async with semaphore:
        image_success = False
        json_success = False
        json_path = filepath.with_suffix('.json')

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)

            
            if filepath.exists():
                image_success = True  
            else:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        async with aiofiles.open(filepath, 'wb') as f:
                            await f.write(content)
                        image_success = True
                    else:
                        logger.warning(f"Failed to download {url}: HTTP {response.status}")

            
            if json_path.exists():
                json_success = True  
            else:
                json_success = await save_json_metadata(filepath, card_json)

            return (image_success, json_success)

        except Exception as e:
            logger.error(f"Error processing {filepath.name}: {e}")
            return (image_success, json_success)


def extract_card_data(card: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract relevant data from a card JSON object"""
    # Skip non-English cards
    if card.get('lang') != 'en':
        return None

    # Skip digital-only cards
    if card.get('digital', False):
        return None

    #  (handle double-faced cards)
    if 'image_uris' in card and 'large' in card['image_uris']:
        image_url = card['image_uris']['large']
    elif 'card_faces' in card and len(card['card_faces']) > 0:
        # For double-faced cards, use the front face
        if 'image_uris' in card['card_faces'][0] and 'large' in card['card_faces'][0]['image_uris']:
            image_url = card['card_faces'][0]['image_uris']['large']
        else:
            return None
    else:
        return None

    return {
        'id': card['id'],
        'name': card['name'],
        'set': card['set'],
        'collector_number': card['collector_number'],
        'image_url': image_url,
        'finishes': card.get('finishes', ['nonfoil']),
        'full_json': card  
    }


async def download_cards(limit: Optional[int] = None, sets: Optional[list] = None, finishes: list = ['nonfoil', 'foil', 'etched']):
    """Main download function"""
    logger.info("Fetching default cards bulk data URL...")

    
    local_json = DATA_DIR.parent / "default-cards.json"
    if local_json.exists():
        logger.info(f"Using local file: {local_json}")
        json_source = str(local_json)
    else:
        json_url = await fetch_bulk_data_info()
        logger.info(f"Downloading from: {json_url}")
        json_source = json_url

    
    download_queue = []
    seen_ids = set()
    skipped_complete = 0  

    logger.info("Parsing card data...")

    def process_card(card_data: Dict[str, Any], finishes_filter: list) -> int:
        """Process a card and add to queue if needed. Returns count of skipped (complete) items."""
        nonlocal skipped_complete
        skipped = 0

        for finish in card_data['finishes']:
            if finish not in finishes_filter:
                continue

            safe_name = card_data['name']
            for char in ['/', '\\', '"', '<', '>', ':', '|', '?', '*']:
                safe_name = safe_name.replace(char, '_')

            
            set_code = card_data['set'].upper()
            if set_code in ['CON', 'PRN', 'AUX', 'NUL'] or \
               (len(set_code) == 4 and set_code[:3] in ['COM', 'LPT'] and set_code[3].isdigit()):
                set_code = set_code + '_'

            filename = f"{card_data['collector_number']}_{safe_name}_{card_data['id'][:12]}.jpg"
            filepath = DATA_DIR / set_code / finish / filename
            json_path = filepath.with_suffix('.json')

            
            if filepath.exists() and json_path.exists():
                skipped += 1
                continue

            
            download_queue.append((card_data['image_url'], filepath, card_data['full_json']))

        return skipped

    
    if json_source.startswith('http'):
        async with aiohttp.ClientSession() as session:
            async with session.get(json_source) as response:
                parser = ijson.items(response.content, 'item')
                async for card in parser:
                    if limit and len(download_queue) >= limit:
                        break

                    card_data = extract_card_data(card)
                    if not card_data:
                        continue

                    
                    if sets and card_data['set'].upper() not in [s.upper() for s in sets]:
                        continue

                    
                    if card_data['id'] in seen_ids:
                        continue
                    seen_ids.add(card_data['id'])

                    skipped_complete += process_card(card_data, finishes)
    else:
        
        with open(json_source, 'rb') as f:
            parser = ijson.items(f, 'item')
            for card in parser:
                if limit and len(download_queue) >= limit:
                    break

                card_data = extract_card_data(card)
                if not card_data:
                    continue

                
                if sets and card_data['set'].upper() not in [s.upper() for s in sets]:
                    continue

                
                if card_data['id'] in seen_ids:
                    continue
                seen_ids.add(card_data['id'])

                skipped_complete += process_card(card_data, finishes)

    logger.info(f"Skipped {skipped_complete} cards (already have image + JSON)")

    if not download_queue:
        logger.info("No new images or JSON files to create")
        return

    logger.info(f"Processing {len(download_queue)} cards (download image and/or create JSON)...")

    
    semaphore = asyncio.Semaphore(CONCURRENT_DOWNLOADS)
    async with aiohttp.ClientSession() as session:
        tasks = [
            download_image_and_json(session, url, filepath, card_json, semaphore)
            for url, filepath, card_json in download_queue
        ]

        
        results = await tqdm_asyncio.gather(*tasks, desc="Processing", unit="cards")

        images_ok = sum(1 for img, _ in results if img)
        jsons_ok = sum(1 for _, js in results if js)
        logger.info(f"Results: {images_ok} images OK, {jsons_ok} JSON files OK")


def main():
    parser = argparse.ArgumentParser(description="Download Scryfall reference card images")
    parser.add_argument('--limit', type=int, help="Limit number of cards to download (for testing)")
    parser.add_argument('--sets', nargs='+', help="Only download specific sets (e.g., M21 NEO)")
    parser.add_argument('--finishes', nargs='+', default=['nonfoil', 'foil', 'etched'],
                       choices=['nonfoil', 'foil', 'etched'],
                       help="Which finishes to download")
    args = parser.parse_args()

    
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    
    asyncio.run(download_cards(
        limit=args.limit,
        sets=args.sets,
        finishes=args.finishes
    ))


if __name__ == "__main__":
    main()