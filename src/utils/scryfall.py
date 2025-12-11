"""
src/utils/scryfall.py: Scryfall metadata utilities
Helper functions for parsing Scryfall data and filenames
"""

import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple


def parse_card_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse card metadata from Scryfall filename
    Common formats:
    - {collector_number}_{name}_{hash}.jpg (e.g., "357_The Mindskinner_e5a967e3b23b.jpg")
    - {name}_{set}_{collector_number}.jpg
    - {set}_{collector_number}.jpg
    - {name}.jpg
    
    Args:
        filename: Image filename
        
    Returns:
        Dictionary with parsed metadata (name, set_code, collector_number) or None
    """
    # Remove extension
    name_part = Path(filename).stem
    
    # First, check if it's the format: {collector_number}_{name}_{hash}
    # Pattern: starts with digits or letter+dash+digits (like "A-130"), then underscore, then name
    # Supports: "357_The Mindskinner_hash" and "A-130_A-Armory Veteran_hash"
    collector_first_pattern = r'^([A-Z]?-?\d+)_(.+?)_([a-f0-9]{12,})$'
    match_collector_first = re.match(collector_first_pattern, name_part, re.IGNORECASE)

    if match_collector_first:
        # Format: collector_number_name_hash
        collector_number = match_collector_first.group(1)
        name_part_clean = match_collector_first.group(2).replace('_', ' ')
        # Don't extract set_code from filename in this format - it should come from directory path
        result = {
            'collector_number': collector_number,
            'name': name_part_clean
        }
        return result
    
    # Try to extract set code and collector number (old format)
    # Pattern: something like "SET123" or "SET-123" or "SET_123"
    # But avoid matching pure numbers at the start (those are collector numbers)
    set_pattern = r'([A-Z]{2,5})[-_]?(\d+)'
    match = re.search(set_pattern, name_part, re.IGNORECASE)
    
    if match:
        set_code = match.group(1).upper()
        collector_number = match.group(2)
        # Extract name (everything before the set code)
        name = name_part[:match.start()].strip('_-').replace('_', ' ').replace('-', ' ')
        if not name:
            name = None
    else:
        # No set/collector pattern found, assume entire stem is name
        name = name_part.replace('_', ' ').replace('-', ' ')
        set_code = None
        collector_number = None
    
    result = {}
    if name:
        result['name'] = name
    if set_code:
        result['set_code'] = set_code
    if collector_number:
        result['collector_number'] = collector_number
    
    return result if result else None


def load_scryfall_json(json_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load Scryfall JSON metadata file
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Parsed JSON data or None
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def detect_variant_type(metadata: Dict[str, Any]) -> str:
    """
    Detect card variant type from Scryfall metadata

    Uses frame_effects and other Scryfall fields to determine variant type.

    Args:
        metadata: Card metadata dict (may contain 'full_json' with Scryfall data)

    Returns:
        Variant type string: 'normal', 'extended_art', 'showcase', 'borderless', etc.
    """
    full_json = metadata.get('full_json', {})

    if not full_json:
        return 'normal'

    frame_effects = full_json.get('frame_effects', [])

    # Check frame effects for variant types
    if 'extendedart' in frame_effects:
        return 'extended_art'
    if 'showcase' in frame_effects:
        return 'showcase'
    if 'borderless' in frame_effects:
        return 'borderless'
    if 'etched' in frame_effects:
        return 'etched'
    if 'fullart' in frame_effects or full_json.get('full_art'):
        return 'full_art'

    # Check promo types
    if full_json.get('promo'):
        promo_types = full_json.get('promo_types', [])
        if 'serialized' in promo_types:
            return 'serialized'
        if 'textured' in promo_types:
            return 'textured'
        if 'stamped' in promo_types:
            return 'stamped'

    return 'normal'


def extract_metadata_from_path(image_path: Path, json_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Extract card metadata from image path and optional JSON file
    
    Args:
        image_path: Path to card image
        json_path: Optional path to Scryfall JSON metadata file
        
    Returns:
        Dictionary with metadata (name, set_code, collector_number, finish, etc.)
    """
    metadata = {}
    
    # Try to parse from filename
    filename_metadata = parse_card_filename(image_path.name)
    if filename_metadata:
        metadata.update(filename_metadata)
    
    # Try to load from JSON if provided
    if json_path and json_path.exists():
        json_data = load_scryfall_json(json_path)
        if json_data:
            # Extract common Scryfall fields
            if 'name' in json_data:
                metadata['name'] = json_data['name']
            if 'set' in json_data:
                metadata['set_code'] = json_data['set'].upper()
            if 'collector_number' in json_data:
                metadata['collector_number'] = json_data['collector_number']
            if 'id' in json_data:
                metadata['scryfall_id'] = json_data['id']
            if 'finishes' in json_data:
                # Determine finish from path or JSON
                if 'foil' in str(image_path).lower() or 'foil' in json_data.get('finishes', []):
                    metadata['finish'] = 'foil'
                else:
                    metadata['finish'] = 'nonfoil'
            # Store full JSON for reference
            metadata['full_json'] = json_data
    
    # Determine finish from path if not in JSON
    if 'finish' not in metadata:
        path_str = str(image_path).lower()
        if 'foil' in path_str:
            metadata['finish'] = 'foil'
        else:
            metadata['finish'] = 'nonfoil'
    
    return metadata

