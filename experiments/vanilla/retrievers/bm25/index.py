from rankify.indexing import BGEIndexer, LuceneIndexer, DPRIndexer, ColBERTIndexer

indexer = LuceneIndexer(corpus_path='../../../index/corpus.jsonl', output_dir='./index',
                        chunk_size=1024, threads=8, index_type="wiki")

indexer.build_index()
indexer.load_index()

print("BM25 indexing complete.")