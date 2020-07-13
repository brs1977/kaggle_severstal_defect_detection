import pandas as pd
from mlcomp.contrib.split import stratified_group_k_fold

import os

import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np

from mlcomp.contrib.transform.rle import rle2mask

#make fold dataframe
df = pd.read_csv('data/train.csv')
df['fold'] = stratified_group_k_fold(label='ClassId', group_column='ImageId', df=df, n_splits=5)
df.to_csv('data/fold.csv', index=False)


#make masks data
df = pd.read_csv('data/fold.csv')
os.makedirs('data/train_masks', exist_ok=True)

df = df.sort_values(by='ImageId')
size = (256, 1600)
mask = np.zeros(size)
res = []
oldImageId = df.ImageId[0]
for row in tqdm(df.itertuples(), total=df.shape[0]):
    pixels = row.EncodedPixels
    if not isinstance(pixels, str):
        pixels = ''

    imageId = row.ImageId    
    mask_cls = rle2mask(pixels, size[::-1])
    mask[mask_cls > 0] = row.ClassId

    if oldImageId != imageId:
        base_name = imageId.split('.')[0]
        cv2.imwrite(f'data/train_masks/{base_name}.png', mask)
        mask = np.zeros(size)
        res.append(
            {
                'fold': row.fold,
                'image': f'{base_name}.jpg',
                'mask': f'{base_name}.png'
            }
        )
    oldImageId = imageId

pd.DataFrame(res).to_csv('data/masks.csv', index=False)