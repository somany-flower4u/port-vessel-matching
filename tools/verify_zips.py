"""verify_zips.py ZIP... : full CRC check (zipfile.testzip, streaming) + entry count/size vs the source package dir."""
import sys, zipfile, os
from pathlib import Path
W = Path(os.environ.get("SEOKCHEON_WS", ".") + "/work")
for z in sys.argv[1:]:
    name = Path(z).name.replace("submit_claude_", "").replace(".zip", "")
    src = W / name
    files = [src / "script.py", src / "requirements.txt"] + sorted(
        p for p in (src / "model").rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store")
    with zipfile.ZipFile(z) as zf:
        infos = zf.infolist()
        size_ok = sum(i.file_size for i in infos) == sum(p.stat().st_size for p in files)
        bad = zf.testzip()
    print(f"{name}: entries {len(infos)}/{len(files)} sizes {'ok' if size_ok else 'MISMATCH'} crc {'ok' if bad is None else 'BAD ' + bad}", flush=True)
