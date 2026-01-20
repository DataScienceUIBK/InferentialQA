import os
from rankify.indexing import BGEIndexer, LuceneIndexer, DPRIndexer, ColBERTIndexer

os.makedirs('./index', exist_ok=True)

JAVA_HOME = "/home/c703/c7031431/java/jdk-21.0.5"
os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["JDK_HOME"] = JAVA_HOME
os.environ["PATH"] = f'{JAVA_HOME}/bin:' + os.environ.get("PATH", "")
ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f'{JAVA_HOME}/lib/server:{JAVA_HOME}/lib:' + ld


indexer = DPRIndexer(corpus_path='../../../index/corpus.jsonl', output_dir="./index",
                        chunk_size=1024, threads=32, batch_size=128, index_type="wiki", retriever_name="dpr")
indexer.build_index()
indexer.load_index()
print("DPR wiki indexing complete.")