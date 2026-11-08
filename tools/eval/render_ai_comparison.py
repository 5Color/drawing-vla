from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "train"))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dataset import decode_png_gray
from brush import render, runs_from_traj

OUT = ROOT / "outputs"
s = json.loads((OUT / "ai_comparison_matched.json").read_text(encoding="utf-8"))
ref = json.loads((ROOT / "data" / "raw" / s["target_source"]).read_text(encoding="utf-8"))
target = decode_png_gray(ref["target_128_png"])
size, header = 320, 76
bg = np.array([250, 248, 243])
mask = np.asarray(Image.fromarray(target).resize((size,size),Image.Resampling.BICUBIC),np.float32)/255
target_img = Image.fromarray((bg[None,None,:]*(1-mask[:,:,None])+np.array([25,25,32])[None,None,:]*mask[:,:,None]).astype(np.uint8))
tiles = [("Same apple target", "All candidates seed 0, exec12 baseline", target_img)]
labels = {"v4_matched_x12":"v4 / execute 12", "v4_matched_te0.01q6":"v4 / TE q6, m=.01", "v5b_matched_x12":"v5b / execute 12", "v5b_matched_te0.01q6":"v5b / TE q6, m=.01", "v5b_matched_te0.01q4":"v5b / TE q4, m=.01  [SELECTED]"}
for row in s["candidates"]:
    r = json.loads(Path(row["rollout_path"]).read_text(encoding="utf-8"))["runs"][0]
    image = np.asarray(render(runs_from_traj(np.asarray(r["executed"],np.float32)),size),np.float32)
    # Pale common target is visible only where rollout ink has not covered it.
    faint = bg[None,None,:]*(1-mask[:,:,None]*.23)+np.array([25,25,32])[None,None,:]*(mask[:,:,None]*.23)
    img = Image.fromarray(np.minimum(image,faint).astype(np.uint8))
    b = row["export_precision_b"]
    subtitle = f"B {b['score']:.1f} | wave {row['smooth']['disp_wave_deg']:.2f} deg | {row['strokes']} strokes"
    tiles.append((labels[row["name"]], subtitle, img))
sheet = Image.new("RGB",(size*3+32, (size+header)*2+24),(20,20,26))
font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf",16)
small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf",13)
for i,(title,subtitle,img) in enumerate(tiles):
    x,y = 8+(i%3)*(size+8), 8+(i//3)*(size+header+8)
    sheet.paste(img,(x,y+header))
    d=ImageDraw.Draw(sheet)
    color=(130,240,180) if "SELECTED" in title else (240,240,240)
    d.text((x+4,y+7),title,fill=color,font=font)
    d.text((x+4,y+33),subtitle,fill=(180,180,190),font=small)
    if i:
        d.text((x+4,y+53),"Target: light gray | prediction: dark brush",fill=(150,150,160),font=small)
sheet.save(OUT / "ai_comparison_brush.png")
print(OUT / "ai_comparison_brush.png")
