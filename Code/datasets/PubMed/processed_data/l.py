import pathlib
import pandas as pd

def build_iterator_from_csv(path_to_csv, batch_size):
    return pd.read_csv(path_to_csv, header=[0], chunksize=batch_size, iterator=True, delimiter="|")

bs = 128
list_paths_to_csv = list(pathlib.Path('./').glob('*.csv'))
print(list_paths_to_csv)

iterator = build_iterator_from_csv(list_paths_to_csv[0], bs)
iterator_it = iter(iterator)

while True:
    batch = next(iterator_it)
    if len(batch) < bs:
        print(batch)

