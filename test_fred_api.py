#!/usr/bin/env python3
"""
Quick test script to validate FRED API key
"""
import sys

# Get API key from command line
if len(sys.argv) < 2:
    print("Usage: python test_fred_api.py YOUR_API_KEY")
    print("\nExample: python test_fred_api.py abcdef1234567890abcdef1234567890")
    sys.exit(1)

api_key = sys.argv[1].strip()

print("=" * 60)
print("FRED API Key Test")
print("=" * 60)
print()

# Test 1: Check key format
print("Test 1: API Key Format")
print(f"  Raw length: {len(sys.argv[1])}")
print(f"  Stripped length: {len(api_key)}")
print(f"  First 4 chars: {api_key[:4] if len(api_key) >= 4 else 'TOO SHORT'}")
print(f"  Last 4 chars: {api_key[-4:] if len(api_key) >= 4 else 'TOO SHORT'}")
print(f"  Has spaces: {' ' in api_key}")
print(f"  Is alphanumeric: {api_key.replace('_', '').isalnum()}")
print()

# Test 2: Try to import fredapi
print("Test 2: Import fredapi library")
try:
    from fredapi import Fred
    print("  ✓ fredapi imported successfully")
except ImportError as e:
    print(f"  ❌ Failed to import fredapi: {e}")
    print("  Install with: pip install fredapi")
    sys.exit(1)
print()

# Test 3: Initialize Fred object
print("Test 3: Initialize FRED connection")
try:
    fred = Fred(api_key=api_key)
    print("  ✓ FRED object created")
except Exception as e:
    print(f"  ❌ Failed to create FRED object: {e}")
    sys.exit(1)
print()

# Test 4: Try simple series fetch (DFF - Fed Funds Rate)
print("Test 4: Fetch test series (DFF - Fed Funds Rate)")
try:
    series = fred.get_series('DFF', observation_start='2024-01-01', observation_end='2024-01-10')
    if series is not None and not series.empty:
        print(f"  ✓ Fetched {len(series)} data points")
        print(f"  Latest value: {series.iloc[-1]:.2f}% (date: {series.index[-1].date()})")
    else:
        print("  ⚠️ Series returned but is empty")
except Exception as e:
    print(f"  ❌ Failed to fetch series: {e}")
    print(f"  Error type: {type(e).__name__}")

    error_str = str(e).lower()
    if 'mismatched tag' in error_str or 'html' in error_str:
        print("\n  💡 This looks like an authentication error")
        print("     - API key may not be activated")
        print("     - Check email for activation link")
    elif '400' in error_str or 'bad request' in error_str:
        print("\n  💡 Bad request - possibly invalid API key format")
    elif '429' in error_str or 'rate limit' in error_str:
        print("\n  💡 Rate limit exceeded - wait a moment and try again")
    else:
        print("\n  💡 Unknown error - check network connection")

    sys.exit(1)
print()

# Test 5: Try macro series (what portfolio app needs)
print("Test 5: Fetch macro series (RRPONTSYD - ON RRP)")
try:
    series = fred.get_series('RRPONTSYD', observation_start='2024-01-01')
    if series is not None and not series.empty:
        print(f"  ✓ Fetched {len(series)} data points")
        print(f"  Latest value: ${series.iloc[-1]/1000:.0f}B (date: {series.index[-1].date()})")
    else:
        print("  ⚠️ Series returned but is empty")
except Exception as e:
    print(f"  ❌ Failed to fetch series: {e}")
    sys.exit(1)
print()

# Test 6: Try multiple series (simulate portfolio app)
print("Test 6: Fetch multiple macro series")
test_series = {
    'RRPONTSYD': 'ON RRP',
    'WTREGEN': 'TGA',
    'WRESBAL': 'Bank Reserves',
    'SOFR': 'SOFR Rate',
    'BAMLH0A0HYM2': 'HY OAS'
}

success_count = 0
failed = []

for code, name in test_series.items():
    try:
        series = fred.get_series(code, observation_start='2024-01-01')
        if series is not None and not series.empty:
            success_count += 1
            print(f"  ✓ {code:15} ({name}): {len(series)} points")
        else:
            failed.append(f"{code} (empty)")
    except Exception as e:
        failed.append(f"{code} ({str(e)[:50]})")

print()
print(f"Results: {success_count}/{len(test_series)} series fetched successfully")

if failed:
    print("\n  Failed series:")
    for f in failed:
        print(f"    - {f}")
print()

# Summary
print("=" * 60)
if success_count == len(test_series):
    print("✅ ALL TESTS PASSED - API key is working correctly!")
    print()
    print("Your API key is valid and can be used in the portfolio app.")
elif success_count > 0:
    print(f"⚠️ PARTIAL SUCCESS - {success_count}/{len(test_series)} series working")
    print()
    print("API key is valid but some series failed.")
    print("This may be due to permissions or temporary issues.")
else:
    print("❌ ALL TESTS FAILED - API key is NOT working")
    print()
    print("Please check:")
    print("1. API key is activated (click email link)")
    print("2. No extra spaces in the key")
    print("3. Network connection is working")
    print("4. FRED website is accessible")

print("=" * 60)
