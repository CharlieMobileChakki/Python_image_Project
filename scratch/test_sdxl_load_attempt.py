import sys
import os
sys.path.insert(0, ".")

print("=== Testing SDXL Model Loading Attempt ===")
from app.experimental.sdxl_reconstruction import GarmentReconstructionEngine

engine = GarmentReconstructionEngine()
print("engine.is_available():", engine.is_available())
print("engine.is_loaded():", engine.is_loaded())
print("engine.configured_device:", engine.configured_device)

try:
    print("\nAttempting engine.load()...")
    engine.load()
    print("SUCCESS: Engine loaded!")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
