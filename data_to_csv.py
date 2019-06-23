import os
import pandas as pd

# iterators over files in respective folders
apples = os.scandir('data/apples')
bananas = os.scandir('data/bananas')

# create a data frame to save locations and labels
df = pd.DataFrame(columns=['file_name', 'label'])

for apple in apples:
    loc = 'data/apples/{}'.format(apple.name)
    df = df.append({'file_name': loc, 'label': 0}, ignore_index=True)

for banana in bananas:
    loc = 'data/bananas/{}'.format(banana.name)
    df = df.append({'file_name': loc, 'label': 1}, ignore_index=True)

# save as csv file
df.to_csv('file_names.csv', header=None, index=False)
