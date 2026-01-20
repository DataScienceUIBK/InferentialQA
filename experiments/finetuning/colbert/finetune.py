import os.path
import shutil
import argparse
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer


def finetune(num_of_passages):
    num_of_epochs = epochs_dict[num_of_passages]
    for i in range(1, num_of_epochs + 1):
        with Run().context(RunConfig(experiment="msmarco", root=f"./{num_of_passages}_positive/checkpoints")):

            config = ColBERTConfig(
                bsize=32, doc_maxlen=256, query_maxlen=64
            )
            trainer = Trainer(
                triples=f"./{num_of_passages}_positive/train_dir/train_data.jsonl",
                collection='./corpus_dir/collection_train.tsv',
                queries=f'./{num_of_passages}_positive/dev_dir/queries_train.tsv',
                config=config,
            )

            if i == 1:
                trainer.train('bert-base-uncased')
            else:
                trainer.train(os.path.abspath(f"./{num_of_passages}_positive/checkpoints/checkpoint_{i - 1}"))

            checkpoint_dir = f'./{num_of_passages}_positive/checkpoints'
            for _ in range(7):
                checkpoint_dir = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])
            shutil.move(checkpoint_dir, f"./{num_of_passages}_positive/checkpoints/checkpoint_{i}")
            shutil.rmtree(f'./{num_of_passages}_positive/checkpoints/msmarco')
    shutil.move(f"./{num_of_passages}_positive/checkpoints/checkpoint_{num_of_epochs}",
                f"./{num_of_passages}_positive/checkpoints/checkpoint_final")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_of_passages', type=int, required=True)

    args = parser.parse_args()
    num_of_passages = int(args.num_of_passages)

    epochs_dict = {1: 10, 5: 5, 10: 2, 50: 1, 100: 1, 200: 1}
    finetune(num_of_passages)
