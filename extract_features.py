import argparse, cv2, pandas as pd
from pathlib import Path
from facial_compassion_project import extract_landmark_features

def main(images: Path, out: Path):        
    rows = []
    for img_path in images.glob("*.[jp][pn]g"):
        feats = extract_landmark_features(cv2.imread(str(img_path)))
        if feats:
            rows.append({"image_id": img_path.name, **feats.__dict__})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} rows to {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True,
                   help="Folder with face images")
    p.add_argument("--out", type=Path,   required=True,
                   help="Destination CSV")
    args = p.parse_args()
    main(args.images, args.out)  
