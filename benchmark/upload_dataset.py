from datasets import load_dataset

dataset = load_dataset('csv', data_files='data.csv')

print(dataset)