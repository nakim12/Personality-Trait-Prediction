import argparse, csv, pathlib, shutil
from lxml import etree

def images_from_xml(xml_path: pathlib.Path):
    root = etree.parse(str(xml_path)).getroot()
    for img in root.xpath(".//image"):
        yield pathlib.Path(img.get("file"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="ImgLab XML file")
    ap.add_argument("--out", required=True, help="Output CSV (e.g. data/my_labels.csv)")
    args = ap.parse_args()

    xml_path = pathlib.Path(args.xml).expanduser().resolve()
    out_csv  = pathlib.Path(args.out).expanduser().resolve()

    IMAGE_FOLDER = xml_path.parent               # where PNGs really live
    raw_dir      = out_csv.parent / "raw_images" # pipeline destination
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows, copied = [], 0
    for rel in images_from_xml(xml_path):
        src = (IMAGE_FOLDER / rel).resolve()
        if not src.exists():
            print("⚠️  Missing:", src); continue

        dst = raw_dir / src.name
        if src != dst:               # copy only if not already there
            shutil.copy(src, dst)
            copied += 1

        rows.append([src.name, 0, "neutral"])   # placeholder

    with out_csv.open("w", newline="") as f:
        csv.writer(f).writerows([["image_id","compassion_pct","class_label"], *rows])

    print(f"✅ Copied {copied} images → {raw_dir}")
    print(f"✅ Wrote  {out_csv}")

if __name__ == "__main__":
    main()
