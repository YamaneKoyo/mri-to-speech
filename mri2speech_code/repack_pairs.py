# repack_pairs.py
import os, glob, shutil
from tqdm import tqdm

def repack_pairs_by_video(pairs_dir: str, dry_run: bool = False):
    assert os.path.isdir(pairs_dir), f"{pairs_dir} not found"
    files = glob.glob(os.path.join(pairs_dir, "*.npz"))
    print(f"[REPACK] scanning: {pairs_dir}  files={len(files)}")
    moved = skipped = 0
    for fp in tqdm(files):
        name = os.path.basename(fp)
        # 蜈磯ｭ3譯√ｒ縲悟虚逕ｻID縲阪→縺励※謇ｱ縺・ｼ・tn縺ｮ蜻ｽ蜷阪↓貅匁侠・・        vid = name.split('_')[0]
        if not vid.isdigit() or len(vid) != 3:
            skipped += 1
            continue
        dst_dir = os.path.join(pairs_dir, vid)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst):
            # 菴募ｺｦ螳溯｡後＠縺ｦ繧０K
            continue
        if not dry_run:
            shutil.move(fp, dst)
        moved += 1
    print(f"[REPACK] moved={moved}, skipped={skipped}, done.")

if __name__ == "__main__":
    # 萓・
    # !python repack_pairs_by_video.py
    pairs_dir = "/content/drive/MyDrive/MRI_to_Speech/processed_rtmri/pairs_ref4"
    repack_pairs_by_video(pairs_dir, dry_run=False)
