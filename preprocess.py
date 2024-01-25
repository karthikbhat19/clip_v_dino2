import pandas as pd

df = pd.read_csv("archive/captions.txt")
processed_dict = {}

for index, row in df.iterrows():
    if row['image'] not in processed_dict:
        processed_dict[row['image']] = []
    processed_dict[row['image']].append(row['caption'])

print(processed_dict['1001773457_577c3a7d70.jpg'])