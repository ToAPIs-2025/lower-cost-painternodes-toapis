"""Optional ToAPIs nodes. Keys come only from the ComfyUI server environment."""
from pathlib import Path
import json
from .unified_client import Client,generate
PRESETS=json.loads(Path(__file__).with_name("presets.json").read_text(encoding="utf-8"))
class ToAPIsImage:
 @classmethod
 def INPUT_TYPES(cls):return {"required":{"prompt":("STRING",{"multiline":True}),"job_id":("STRING",{"default":"reviewed-image-01"}),"live":("BOOLEAN",{"default":False})}}
 RETURN_TYPES=("IMAGE","STRING")
 FUNCTION="run"
 CATEGORY="ToAPIs"
 def run(self,prompt,job_id,live):
  p=PRESETS["gpt-image-2-vip"];r=generate(Client(),p,prompt,max_usd=".10",live=live,run_id=job_id)
  if not live:raise ValueError("Quote only: "+json.dumps(r)+". Enable live only after review.")
  if r.get("status")!="completed":raise ValueError("Generation incomplete; reconcile saved task")
  import numpy as np,torch
  from PIL import Image
  pixels=np.asarray(Image.open(r["output"]).convert("RGB"),dtype=np.float32)/255.0
  return (torch.from_numpy(pixels)[None,...],json.dumps(r["cost"]))
class ToAPIsVideoPath:
 @classmethod
 def INPUT_TYPES(cls):return {"required":{"prompt":("STRING",{"multiline":True}),"job_id":("STRING",{"default":"reviewed-video-01"}),"live":("BOOLEAN",{"default":False})}}
 RETURN_TYPES=("STRING","STRING")
 FUNCTION="run"
 CATEGORY="ToAPIs"
 def run(self,prompt,job_id,live):
  r=generate(Client(),PRESETS["seedance-2-fast"],prompt,max_usd=".75",live=live,run_id=job_id)
  if not live:raise ValueError("Quote only: "+json.dumps(r)+". Enable live only after review.")
  if r.get("status")!="completed":raise ValueError("Generation incomplete; reconcile saved task")
  return (r["output"],json.dumps(r["cost"]))
NODE_CLASS_MAPPINGS={"ToAPIsImage":ToAPIsImage,"ToAPIsVideoPath":ToAPIsVideoPath}
NODE_DISPLAY_NAME_MAPPINGS={"ToAPIsImage":"ToAPIs Image (opt-in)","ToAPIsVideoPath":"ToAPIs Video Path (opt-in)"}
