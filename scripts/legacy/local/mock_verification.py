
import sys
from unittest.mock import MagicMock

# Mock webdataset before import
sys.modules["webdataset"] = MagicMock()

# Add project root
sys.path.append(".")

def verify_syntax():
    print("Verifying ViCocktail Preprocessor Logic/Syntax...")
    try:
        from src.data.preprocessors.vicocktail import ViCocktailPreprocessor
        print("✅ Import Successful (Dependencies Mocked)")
        
        # Instantiate (Mocking BasePreprocessor init if needed, but it should be fine)
        # BasePreprocessor.__init__ sets data_root and use_precropped
        processor = ViCocktailPreprocessor(data_root="dummy", use_precropped=False)
        print(f"✅ Class Instantiation Successful: {processor}")
        
        if hasattr(processor, 'run') and hasattr(processor, 'collect_metadata'):
            print("✅ Method Structure Valid (run, collect_metadata present)")
        else:
            print("❌ Missing required methods")
            
    except ImportError as e:
        print(f"❌ Import Failed: {e}")
    except Exception as e:
        print(f"❌ Instantiation Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_syntax()
