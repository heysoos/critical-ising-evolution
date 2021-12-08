import pickle
import gzip

def compressed_pickle(title, data):
    with gzip.GzipFile(title + '.pgz', 'w') as f:
        pickle.dump(data, f)