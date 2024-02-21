import arff
import pandas as pd

def read_arff_file(file_path):
  with open(file_path, 'r') as f:
    data = arff.load(f)
  return data

def to_pandas_dataframe(data):
  attributes = [attr[0] for attr in data['attributes']]
  return pd.DataFrame(data['data'], columns=attributes)