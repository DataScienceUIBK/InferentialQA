from rankify.indexing import BGEIndexer, LuceneIndexer, DPRIndexer, ColBERTIndexer, ContrieverIndexer

indexer = ContrieverIndexer(
    corpus_path='../../../index/corpus.jsonl',
    output_dir="index",
    index_type="wiki",
    encoder_name="facebook/contriever",
    batch_size=128
)

indexer.build_index()
indexer.load_index()

print("Contriever wiki indexing complete.")