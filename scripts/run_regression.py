#!/usr/bin/env python
"""
Regression Test Runner
Run recognition on a set of "golden" images and verify they match expected results.
Usage: uv run python scripts/run_regression.py
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import contextlib
import os
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.recognition.matcher import CardMatcher

# Directory configuration
REGRESSION_DIR = project_root / "tests" / "regression"
EXPECTED_FILE = REGRESSION_DIR / "expected.json"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache matchers to avoid reloading indices
_matcher_cache = {}

def get_matcher(set_code: str) -> CardMatcher:
    """Get or create a CardMatcher for the given set"""
    if set_code not in _matcher_cache:
        print(f"Loading matcher for set {set_code}...")
        _matcher_cache[set_code] = CardMatcher(set_code=set_code)
    return _matcher_cache[set_code]

@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def load_expected_results() -> Dict[str, Any]:
    """Load expected results from JSON"""
    if not EXPECTED_FILE.exists():
        return {}
    with open(EXPECTED_FILE, 'r') as f:
        return json.load(f)

def save_expected_results(results: Dict[str, Any]):
    """Save results as new expected values"""
    with open(EXPECTED_FILE, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"Saved {len(results)} results to {EXPECTED_FILE}")

def run_regression(update_baseline: bool = False):
    """Run regression tests"""
    if not REGRESSION_DIR.exists():
        print(f"Error: Regression directory not found: {REGRESSION_DIR}")
        print("Create it and add some test images!")
        return

    image_files = list(REGRESSION_DIR.glob("*.jpg")) + list(REGRESSION_DIR.glob("*.png"))
    if not image_files:
        print(f"No images found in {REGRESSION_DIR}")
        return

    print(f"Found {len(image_files)} test images.")
    
    expected = load_expected_results()
    current_results = {}
    
    passed = 0
    warnings = 0
    failed = 0
    skipped = 0
    
    print("\n" + "="*80)
    print(f"{'FILENAME':<30} | {'STATUS':<10} | {'CONFIDENCE':<10} | {'EXPECTED':<20} | {'ACTUAL':<20}")
    print("-" * 80)

    for img_path in image_files:
        filename = img_path.name
        
        # Determine set code
        set_code = None
        if filename in expected:
            set_code = expected[filename].get('set_code')
        
        if not set_code:
            # Try to infer from filename or skip
            # For now, we skip if we don't know the set, as CardMatcher requires it
            print(f"{filename:<30} | SKIPPED    | N/A        | Unknown Set          | N/A")
            skipped += 1
            continue

        try:
            matcher = get_matcher(set_code)
            
            # Run recognition
            # We use suppress_stdout to keep the regression output clean, 
            # but CardMatcher logs to logging system which might still show up depending on config
            with suppress_stdout():
                result = matcher.match_scanned(img_path, use_ocr=True)
                
        except Exception as e:
            print(f"{filename:<30} | ERROR      | N/A        | Error: {str(e)}")
            failed += 1
            continue

        if not result.match_card_id:
            print(f"{filename:<30} | NO MATCH   | 0.0%       | N/A                  | None")
            failed += 1
            continue

        # Get top candidate details
        top_candidate = result.candidates[0]
        match_name = top_candidate.card_name
        confidence = result.confidence
        
        # Store for baseline update
        current_results[filename] = {
            "card_name": match_name,
            "set_code": top_candidate.set_code,
            "collector_number": top_candidate.collector_number,
            "confidence": confidence
        }

        # Compare with expected
        if filename in expected:
            exp = expected[filename]
            exp_name = exp.get('card_name')
            
            if match_name == exp_name:
                if confidence >= 0.90:
                    status = "PASS"
                    passed += 1
                else:
                    status = "WARN"
                    warnings += 1
                
                # Check for confidence regression (>5% drop)
                prev_conf = exp.get('confidence', 0)
                if prev_conf > 0 and (prev_conf - confidence) > 0.05:
                    status = "REGR"
                    warnings += 1
            else:
                status = "FAIL"
                failed += 1
        else:
            status = "NEW"
            warnings += 1
            exp_name = "N/A"

        print(f"{filename:<30} | {status:<10} | {confidence:.1%}      | {str(exp_name)[:20]:<20} | {str(match_name)[:20]:<20}")

    print("=" * 80)
    print(f"Summary: {passed} Passed, {warnings} Warnings, {failed} Failed, {skipped} Skipped")
    
    if update_baseline:
        save_expected_results(current_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run regression tests")
    parser.add_argument("--update", action="store_true", help="Update baseline expected results")
    args = parser.parse_args()
    
    run_regression(update_baseline=args.update)
