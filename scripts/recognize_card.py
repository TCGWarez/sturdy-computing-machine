#!/usr/bin/env python
"""
CLI wrapper for CardMatcher - provides verbose output and debugging
Usage: uv run python scripts/recognize_card.py <image_path> --set <SET> [--finish nonfoil]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.recognition.matcher import CardMatcher

def main():
    parser = argparse.ArgumentParser(
        description='MTG Card Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/recognize_card.py scan.jpg --set SNC
  python scripts/recognize_card.py scan.jpg --set DSK --finish foil
  python scripts/recognize_card.py scan.jpg --set TLA --device cuda
        """
    )
    
    parser.add_argument('image_path', type=Path, help='Path to scanned card image')
    parser.add_argument('--set', type=str, required=True, help='Set code (e.g., TLA, DMR, SLD)')
    parser.add_argument('--finish', type=str, default='nonfoil', choices=['nonfoil', 'foil'], 
                       help='Finish type (default: nonfoil)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use for CLIP model (default: cpu)')
    parser.add_argument('--use-ocr', action='store_true', 
                       help='Enable OCR for disambiguation (slower)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.INFO if not args.debug else logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not args.image_path.exists():
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    print(f"Recognizing: {args.image_path}")
    print(f"Set: {args.set}, Finish: {args.finish}, Device: {args.device}")
    print("=" * 80)
    
    try:
        # Initialize matcher
        print(f"\nInitializing CardMatcher for {args.set}/{args.finish}...")
        matcher = CardMatcher(
            set_code=args.set,
            finish=args.finish,
            device=args.device
        )
        
        # Run recognition
        print(f"Processing image...")
        result = matcher.match_scanned(
            args.image_path,
            use_ocr=args.use_ocr,
            debug=args.debug
        )
        
        # Display results
        print("\n" + "=" * 80)
        print("RESULT:")
        print("=" * 80)
        
        if result.match_card_id and result.candidates:
            top = result.candidates[0]
            
            # Confidence indicator (use ASCII-safe characters for Windows)
            if result.confidence >= 0.91:
                status = "[OK] HIGH CONFIDENCE"
            elif result.confidence >= 0.88:
                status = "[??] MANUAL REVIEW"
            else:
                status = "[!!] LOW CONFIDENCE"
            
            print(f"\n{status} ({result.confidence:.1%})")
            print(f"\nCard: {top.card_name}")
            print(f"Set: {top.set_code} #{top.collector_number}")
            print(f"Finish: {top.finish}")
            print(f"Card ID: {top.card_id}")
            
            print(f"\nScore Breakdown:")
            print(f"  Embedding: {top.embedding_score:.3f}")
            print(f"  pHash:     {top.phash_score:.3f}")
            print(f"  Combined:  {top.combined_score:.3f}")

            if result.confidence < 0.91 and len(result.candidates) > 1:
                print(f"\nAlternative Matches:")
                for i, cand in enumerate(result.candidates[1:4], 2):
                    print(f"  #{i}: {cand.card_name} ({cand.set_code} #{cand.collector_number}) - {cand.combined_score:.1%}")
            
            print(f"\nProcessing time: {result.processing_time:.2f}s")
            
        else:
            print("\n[!!] NO MATCH FOUND")
            print(f"Method: {result.match_method}")

        print("=" * 80 + "\n")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print(f"\nPlease build the index first:")
        print(f"  python scripts/index_set.py {args.set} --finish {args.finish}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
