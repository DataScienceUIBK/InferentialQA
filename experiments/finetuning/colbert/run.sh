clear

python make_colbert_inputs.py

python finetune.py --num_of_passages 1
python finetune.py --num_of_passages 5
python finetune.py --num_of_passages 10
python finetune.py --num_of_passages 50
python finetune.py --num_of_passages 100
python finetune.py --num_of_passages 200

module load cuda
python retrieve.py --num_of_passages 1
python retrieve.py --num_of_passages 5
python retrieve.py --num_of_passages 10
python retrieve.py --num_of_passages 50
python retrieve.py --num_of_passages 100
python retrieve.py --num_of_passages 200