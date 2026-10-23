"""Read a completed local asset; no application billing is invented."""
from pathlib import Path
import base64
def image_fragment(path):
 raw=Path(path).read_bytes()
 if not raw.startswith(b"\x89PNG\r\n\x1a\n"):raise ValueError("Expected verified PNG")
 return {"images":[{"data":base64.b64encode(raw).decode(),"mimeType":"image/png"}]}
def local_asset(path):
 p=Path(path).resolve()
 if not p.is_file():raise ValueError("Missing asset")
 return {"path":str(p),"bytes":p.stat().st_size}
