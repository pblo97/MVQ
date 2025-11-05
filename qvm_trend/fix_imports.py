import re, os

FILES = ["scoring.py", "fundamentals.py", "data_io.py", "app_streamlit.py"]

REPLACERS = [
    # quitar imports relativos (punto) fuera de paquete
    (r"from\s+\.\s*factors\s+import",            "from factors import"),
    (r"from\s+\.\s*factors_growth_aware\s+import","from factors_growth_aware import"),
    (r"from\s+\.\s*data_io\s+import",            "from data_io import"),
    (r"from\s+\.\s*fundamentals\s+import",       "from fundamentals import"),
    # quitar paquete qvm_trend
    (r"from\s+qvm_trend\.",                      "from "),
]

changed = []
for fname in FILES:
    if not os.path.exists(fname):
        continue
    with open(fname, "r", encoding="utf-8") as f:
        txt = f.read()
    new = txt
    for pat, repl in REPLACERS:
        new = re.sub(pat, repl, new)
    if new != txt:
        with open(fname, "w", encoding="utf-8") as f:
            f.write(new)
        changed.append(fname)

print("Archivos modificados:", changed)

# Verificación: listar si queda algún import relativo
leftovers = []
for root, _, files in os.walk("."):
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                s = fh.read()
            if re.search(r"from\s+\.", s):
                leftovers.append(path)

print("Imports relativos restantes:", leftovers)