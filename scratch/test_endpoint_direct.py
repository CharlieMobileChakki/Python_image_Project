import asyncio
import sys
sys.path.insert(0, ".")

print("1. Starting test...", flush=True)
from app.api.reconstruction import reconstruct_garment_endpoint
print("2. Imported reconstruct_garment_endpoint", flush=True)

async def test():
    print("3. Calling reconstruct_garment_endpoint...", flush=True)
    try:
        await reconstruct_garment_endpoint(
            original_image=None,  # Not accessed because is_available check is first
            damaged_garment=None,
            repair_mask=None,
        )
    except Exception as exc:
        print(f"4. Caught expected exception: {type(exc).__name__}", flush=True)
        print(f"   Status Code: {getattr(exc, 'status_code', None)}", flush=True)
        print(f"   Detail: {getattr(exc, 'detail', str(exc))}", flush=True)

asyncio.run(test())
print("5. Done!", flush=True)
