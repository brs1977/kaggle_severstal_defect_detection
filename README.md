# kaggle_severstal_defect_detection

1. Load data
./data/load_data.sh

2. Preprocess images for train
python preprocess.py

3. Train data
./train.sh

4. Trace model
catalyst-dl trace ./logs/{model_path}

