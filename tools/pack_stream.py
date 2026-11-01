"""pack.py with constant memory: files are streamed into the zip (16 MB chunks) and the sha256 is computed by
streaming the output.  Same entry names, order, timestamps, attributes and DEFLATE compression as pack.py;
info.file_size is set before writing so zip64 is chosen exactly as zipfile.writestr would."""
import hashlib, shutil, sys, zipfile
from pathlib import Path
sub, out = Path(sys.argv[1]), Path(sys.argv[2])
files = [sub / "script.py", sub / "requirements.txt"]
files += sorted(p for p in (sub / "model").rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in files:
        info = zipfile.ZipInfo(str(p.relative_to(sub)), date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        info.file_size = p.stat().st_size
        with open(p, "rb") as src, zf.open(info, "w") as dst:
            shutil.copyfileobj(src, dst, 16 << 20)
with zipfile.ZipFile(out) as zf:
    names = zf.namelist()
assert sorted({n.split("/")[0] for n in names}) == ["model", "requirements.txt", "script.py"]
h = hashlib.sha256()
with open(out, "rb") as f:
    for chunk in iter(lambda: f.read(16 << 20), b""):
        h.update(chunk)
print(out, out.stat().st_size, "bytes,", len(names), "entries, sha256", h.hexdigest())
