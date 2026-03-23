import argparse
import sys
import os

# Ensure src module is visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessors.vicocktail import ViCocktailPreprocessor

def main():
    parser = argparse.ArgumentParser(description="Vicocktail 2-Phase Preprocessing Pipeline")
    
    parser.add_argument("--mode", type=str, required=True, choices=['crop', 'extract'], 
                        help="Select Phase: 'crop' (Phase 1) or 'extract' (Phase 2)")
    
    # Paths
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Input directory. For 'crop', this is RAW VIDEO folder. For 'extract', this is CROPPED VIDEO folder.")
    
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Output directory. For 'crop', this is Clean Video folder. For 'extract', this is .pt Feature folder.")
    
    parser.add_argument("--manifest_path", type=str, default="data/manifests/train.jsonl",
                        help="Output path for the manifest file (Only used in 'extract' mode)")

    args = parser.parse_args()

    print("=" * 60)
    print("🍹 VICOCKTAIL PIPELINE")
    print("=" * 60)

    if args.mode == "crop":
        print(f"PHASE 1: CROPPING & CLEANING")
        print(f"RAW Input:   {args.input_dir}")
        print(f"CLEAN Output:{args.output_dir}")
        print("-" * 30)
        
        # Initialize Preprocessor pointing to Raw Data
        preprocessor = ViCocktailPreprocessor(data_root=args.input_dir)
        
        # Run Phase 1
        preprocessor.phase1_crop_dataset(save_dir=args.output_dir)
        
        print("-" * 30)
        print(f"✅ Phase 1 Complete! Clean dataset saved to: {args.output_dir}")
        print(f"   Next, run Phase 2 using this output dir as input.")

    elif args.mode == "extract":
        print(f"PHASE 2: FEATURE EXTRACTION (Whisper + ResNet)")
        print(f"CLEAN Input: {args.input_dir}")
        print(f"FEATURE Out: {args.output_dir}")
        print(f"Manifest:    {args.manifest_path}")
        print("-" * 30)
        
        # Initialize
        preprocessor = ViCocktailPreprocessor(data_root=args.input_dir, use_precropped=True)
        
        # Run Phase 2
        preprocessor.run(
            output_manifest=args.manifest_path,
            output_dir=args.output_dir,
            extract_features=True
        )
        
        print("-" * 30)
        print(f"✅ Phase 2 Complete! Ready for training.")

if __name__ == "__main__":
    main()
