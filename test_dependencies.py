#!/usr/bin/env python3
"""
Test script para verificar que todas las dependencias están instaladas correctamente.
"""
import sys

def test_imports():
    """Test all required imports"""
    results = {}

    # Core
    packages = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',

        # Data & API
        'requests': 'requests',
        'urllib3': 'urllib3',
        'fredapi': 'fredapi',

        # Math & Stats
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'statsmodels': 'statsmodels.api',
        'arch': 'arch',

        # Visualization
        'matplotlib': 'matplotlib.pyplot',
        'seaborn': 'seaborn',
        'plotly': 'plotly.express',

        # Utils
        'tqdm': 'tqdm',
        'pyarrow': 'pyarrow',
        'github': 'PyGithub'
    }

    print("Testing imports...")
    print("-" * 60)

    for display_name, import_name in packages.items():
        try:
            __import__(import_name)
            results[display_name] = '✓'
            print(f"✓ {display_name:20} OK")
        except ImportError as e:
            results[display_name] = '✗'
            print(f"✗ {display_name:20} FAILED: {e}")

    print("-" * 60)

    # Summary
    passed = sum(1 for v in results.values() if v == '✓')
    total = len(results)

    print(f"\nResults: {passed}/{total} packages OK")

    if passed == total:
        print("🎉 All dependencies installed successfully!")
        return 0
    else:
        print(f"⚠️  {total - passed} package(s) missing")
        return 1

if __name__ == "__main__":
    sys.exit(test_imports())
