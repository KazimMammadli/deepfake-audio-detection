
import numpy as np
from src.data.dataset import FoRDataset
from src.features.mel_spectrogram import set_preset

set_preset('for-2sec')

for split in ['training', 'validation', 'testing']:
    print(f'Processing {split}...')
    ds = FoRDataset('../for-2seconds', split=split)
    
    X, y = [], []
    for i in range(len(ds)):
        mel, label = ds.get_item(i)
        X.append(mel)
        y.append(label)
        if i % 1000 == 0:
            print(f'  {i}/{len(ds)}')
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]
    
    np.save(f'data/processed/{split}_X.npy', X)
    np.save(f'data/processed/{split}_y.npy', y)
    print(f'{split}: {X.shape} saved ✅')
