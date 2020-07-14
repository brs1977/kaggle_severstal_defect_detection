# kaggle_severstal_defect_detection

Kaggle competition predicting the location and type of defects found in steel manufacturing.

1. Load data

./data/load_data.sh

2. Preprocess images for train

python preprocess.py

3. Train data

./train.sh

4. Trace model

catalyst-dl trace ./logs/{model_path}

5. Predict

python predictor.py -i {file_image} -o {output_path} -m {file_model}

6. Start REST Server 

uvicorn server:app