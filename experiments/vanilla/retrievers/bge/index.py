from rankify.indexing import BGEIndexer, LuceneIndexer, DPRIndexer, ColBERTIndexer

indexer = BGEIndexer(
    corpus_path='../../../index/corpus.jsonl',
    output_dir="index",
    index_type="wiki",
    encoder_name="BAAI/bge-large-en-v1.5",
    batch_size=128
)

indexer.build_index()
indexer.load_index()

print("BGE wiki indexing complete.")