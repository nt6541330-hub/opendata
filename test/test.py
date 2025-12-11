from pymongo import MongoClient, errors

# ========== 1. 配置区域（按你实际情况修改） ==========

# 如果是本机无账号密码，大概率是这个：
# MONGO_URI = "mongodb://127.0.0.1:27017"

# 如果有账号密码就用这个形式：
# MONGO_URI = "mongodb://username:password@127.0.0.1:27017/?authSource=admin"

MONGO_URI = "mongodb://root:123456@39.104.200.88:41004/?authSource=admin"  # TODO：改成你自己的

SOURCE_DB_NAME = "source_data"
TARGET_DB_NAME = "NEWS"

BATCH_SIZE = 1000  # 每次批量写入多少条，可根据数据量和内存酌情调整


# ========== 2. 公共函数：按批复制集合，遇到重复 _id 跳过 ==========

def copy_collection_append(client, src_db_name, dst_db_name,
                           src_coll_name, dst_coll_name=None,
                           batch_size=1000):
    """
    从 src_db.src_coll_name 读取数据，追加到 dst_db.dst_coll_name
    - 相同 _id：忽略（不覆盖原有数据）
    - 不同 _id：插入
    """
    if dst_coll_name is None:
        dst_coll_name = src_coll_name

    src_db = client[src_db_name]
    dst_db = client[dst_db_name]

    src_coll = src_db[src_coll_name]
    dst_coll = dst_db[dst_coll_name]

    print(f"Start sync: {src_db_name}.{src_coll_name}  ->  {dst_db_name}.{dst_coll_name}")

    cursor = src_coll.find({}, no_cursor_timeout=True)
    total_read = 0
    total_inserted = 0
    batch = []

    try:
        for doc in cursor:
            batch.append(doc)
            total_read += 1

            if len(batch) >= batch_size:
                inserted = _insert_batch_ignore_dup(dst_coll, batch)
                total_inserted += inserted
                batch = []
                print(f"  processed: {total_read} docs, inserted: {total_inserted} docs")

        # 最后一批
        if batch:
            inserted = _insert_batch_ignore_dup(dst_coll, batch)
            total_inserted += inserted
            print(f"  processed: {total_read} docs, inserted: {total_inserted} docs")

    finally:
        cursor.close()

    print(f"Done: {src_db_name}.{src_coll_name} -> {dst_db_name}.{dst_coll_name}, "
          f"read={total_read}, inserted={total_inserted}")
    print("-" * 60)


def _insert_batch_ignore_dup(dst_coll, batch):
    """批量插入，忽略 duplicate key (_id 冲突) 错误"""
    try:
        dst_coll.insert_many(batch, ordered=False)
        return len(batch)
    except errors.BulkWriteError as bwe:
        # 检查是不是除了 11000 以外的错误
        write_errors = bwe.details.get("writeErrors", [])
        for err in write_errors:
            if err.get("code") != 11000:
                # 如果有别的错误就抛出，让你看到问题
                raise

        # 只有重复键错误，说明部分/全部已经存在，忽略即可
        n_inserted = bwe.details.get("nInserted", 0)
        return n_inserted


# ========== 3. 主逻辑：具体要同步哪些集合 ==========

def main():
    client = MongoClient(MONGO_URI)

    # 3.1 业务集合：source_cctv
    copy_collection_append(
        client,
        SOURCE_DB_NAME,
        TARGET_DB_NAME,
        "extract_element_event",
        "extract_element_event",
        batch_size=BATCH_SIZE,
    )

    # # 3.2 GridFS 的 files 集合：fs_cctv.files
    # # 如果你确实用 GridFS 存 CCTV 图片，这两行要保留
    # # 如果图片其实是外部对象存储，只需要上面的 source_cctv 同步即可，把下面两段注释掉
    # copy_collection_append(
    #     client,
    #     SOURCE_DB_NAME,
    #     TARGET_DB_NAME,
    #     "fs_xinhua.files",
    #     "fs_xinhua.files",
    #     batch_size=BATCH_SIZE,
    # )
    #
    # # 3.3 GridFS 的 chunks 集合：fs_cctv.chunks
    # copy_collection_append(
    #     client,
    #     SOURCE_DB_NAME,
    #     TARGET_DB_NAME,
    #     "fs_xinhua.chunks",
    #     "fs_xinhua.chunks",
    #     batch_size=BATCH_SIZE,
    # )

    client.close()


if __name__ == "__main__":
    main()
