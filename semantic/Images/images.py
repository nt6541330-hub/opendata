# semantic/Images/images.py
# -*- coding: utf-8 -*-

import io
from datetime import datetime, timezone
from pymongo import MongoClient
import gridfs
from PIL import Image

# 【修改点 1】引入 settings
from config.settings import settings
from common.utils import sniff_mime, has_alpha

# MIME 映射常量 (保留在本地或移至 settings 均可，此处保留在本地)
MIME_TO_EXT = {
    'image/jpeg': 'jpg',
    'image/png': 'png',
    'image/gif': 'gif',
    'image/bmp': 'bmp',
    'image/webp': 'webp',
    'image/tiff': 'tiff',
    'image/x-icon': 'ico',
}


def main():
    print(f"[Images] 开始图片处理 (Mode: {settings.IMAGES_MODE})...")

    # 【修改点 2】使用 settings 连接数据库
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB_NAME]

    # 【修改点 3】遍历配置中的所有图片桶
    for bucket_name in settings.IMAGES_TARGET_BUCKETS:
        print(f"   -> Scanning bucket: {bucket_name}")
        fs = gridfs.GridFS(db, collection=bucket_name)
        files_col = db[f"{bucket_name}.files"]

        # 构建查询
        q = {}
        if settings.IMAGES_NAME_FILTER:
            q['filename'] = {'$regex': settings.IMAGES_NAME_FILTER}

        cursor = files_col.find(q).sort('uploadDate', 1)
        if settings.IMAGES_LIMIT > 0:
            cursor = cursor.limit(settings.IMAGES_LIMIT)

        count = 0
        changed = 0
        converted = 0

        for doc in cursor:
            count += 1
            fid = doc['_id']
            filename = doc.get('filename') or str(fid)
            contentType = doc.get('contentType')

            try:
                gout = fs.get(fid)
            except Exception as e:
                # print(f"[SKIP] {fid} get error: {e}")
                continue

            # 读取头部嗅探类型
            head = gout.read(16)
            gout.seek(0)
            sniffed = sniff_mime(head) or contentType
            if not sniffed:
                continue

            ext = MIME_TO_EXT.get(sniffed)

            # --- 模式 A: 修复元数据 ---
            if settings.IMAGES_MODE == "fix_metadata":
                set_fields = {}
                if contentType != sniffed:
                    set_fields['contentType'] = sniffed

                lower = filename.lower()
                known_exts = tuple("." + e for e in MIME_TO_EXT.values())
                if not lower.endswith(known_exts) and ext:
                    new_name = f"{filename}.{ext}"
                    set_fields['filename'] = new_name

                if set_fields:
                    changed += 1
                    # print(f"[FIX] {fid} {filename} -> {set_fields}")
                    if not settings.IMAGES_DRY_RUN:
                        files_col.update_one({'_id': fid}, {'$set': set_fields})

            # --- 模式 B: 格式转换 ---
            elif settings.IMAGES_MODE == "convert":
                if sniffed not in MIME_TO_EXT:
                    continue

                try:
                    img = Image.open(gout)
                except Exception:
                    continue

                if getattr(img, "is_animated", False):
                    continue

                tgt_fmt = settings.IMAGES_STANDARD_FORMAT
                if tgt_fmt == "AUTO":
                    tgt_fmt = "PNG" if has_alpha(img) else "JPEG"

                gout.seek(0)
                img = Image.open(gout)

                # 如果转 JPEG 且原图不是 RGB/L (例如 RGBA)，需转换
                if tgt_fmt == "JPEG" and img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')

                buf = io.BytesIO()
                save_kwargs = {}
                if tgt_fmt == "JPEG":
                    save_kwargs.update(quality=92, optimize=True)

                try:
                    img.save(buf, format=tgt_fmt, **save_kwargs)
                except Exception:
                    continue

                data = buf.getvalue()

                new_ext = 'jpg' if tgt_fmt == 'JPEG' else 'png'
                base = filename.rsplit('.', 1)[0]
                new_name = f"{base}.{new_ext}"
                new_ct = 'image/jpeg' if tgt_fmt == 'JPEG' else 'image/png'

                meta = {
                    'original_id': str(fid),
                    'converted_from': sniffed,
                    'converted_at': datetime.now(timezone.utc).isoformat(),
                    'mode': img.mode,
                    'size': img.size,
                }

                # print(f"[PUT] {filename} -> {new_name}")
                if not settings.IMAGES_DRY_RUN:
                    # 存入新文件
                    fs.put(data, filename=new_name, contentType=new_ct, metadata=meta)
                    converted += 1

                    # 可选：删除旧文件
                    if settings.IMAGES_DELETE_OLD:
                        try:
                            fs.delete(fid)
                        except Exception:
                            pass
            else:
                pass

        print(f"   -> Bucket {bucket_name} 完成: scanned={count}, fixed={changed}, converted={converted}")

    print("[Images] 所有处理结束")


if __name__ == "__main__":
    main()