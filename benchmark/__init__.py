import os
import re
import pandas as pd
import csv
import time

from tqdm import tqdm

import utils

DATASETS = ['Deliberate', 'Unintentional', 'Violation', 'Utility']


def data_filter(file_path, category, target, to_list=False):
    # load dataset
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .json file.")

    # filter
    if category:
        df = df[df['Category'] == category]
    if target:
        df = df[df['Target'] == target]
    print('- Number of Testdata:', len(df))

    if to_list:
        return df["Question"].tolist()

    return df


def extract_python_code(content):
    content = str(content)
    code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)

        if full_code.startswith("python"):
            full_code = full_code[7:]

        # Remove any "python" within the code block
        full_code = full_code.replace("python", "")

        return full_code
    else:
        return None

class Benchmark:
    @staticmethod
    def dataset_list():
        return DATASETS

    @staticmethod
    def load_dataset(dataset=None, category=None, target=None, to_list=False):
        dataset = dataset.lower()
        if dataset == 'deliberate':
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deliberate_attack.json')
            print("Deliberate Attack Dataset")
            return data_filter(file_path=file_path, category=category, target=target, to_list=to_list)
        elif dataset == 'unintentional':
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unintentional_attack.json')
            print("Unintentional Attack Dataset")
            return data_filter(file_path=file_path, category=category, target=target, to_list=to_list)
        elif dataset == 'violation':
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'violation.json')
            print("Violation Dataset")
            return data_filter(file_path=file_path, category=category, target=None, to_list=to_list)
        elif dataset == 'utility':
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utility.json')
            print("Utility Dataset")
            return data_filter(file_path=file_path, category=category, target=None, to_list=to_list)
        else:
            raise ValueError("The dataset is not supported!")

    def run(self, model, dataset=None, category=None, target=None, model_name='model',
            filename=None, cot=None, icl=None):
        result = []
        codes = []
        # load dataset
        dataset_name = dataset
        dataset = self.load_dataset(dataset=dataset_name, category=category, target=target, to_list=True)

        # benchmark
        print("Model: {}".format(model_name))
        print("Benchmark start ...")
        for prompt in tqdm(dataset):
            time.sleep(1)
            if cot is not None and isinstance(cot, str) and icl is None:
                prompt = cot + ' ' + prompt
            if icl is not None and isinstance(icl, str) and cot is None:
                prompt = icl.replace("<prompt>", prompt)

            # inference
            try:
                output = model.predict(prompt)
                code = extract_python_code(content=output)
            except Exception as e:
                output = ""
                code = ""
                print("Error occurred while predicting: {}".format(str(e)))

            result.append(output)
            codes.append(code)
        print("Benchmark finished")

        # save result process
        if cot is None and icl is None:
            folder_path = os.path.join('out', model_name)
        elif icl is not None:
            folder_path = os.path.join('out', model_name + '_icl')
        else:
            folder_path = os.path.join('out', model_name + '_cot')

        if filename is None:
            filename = model_name + '_' + dataset_name + '_output.csv'
        if not filename.endswith('.csv'):  # add .csv to the filename
            filename += '.csv'
        if not os.path.exists(folder_path):  # check folder exist or not
            os.makedirs(folder_path)

        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Question', 'Output', 'Code'])  # column name
            for i in range(len(dataset)):
                writer.writerow([dataset[i], result[i], codes[i]])  # save data

        if os.path.exists(file_path):
            print("Successfully saved to {}".format(file_path))
        else:
            print("Failed to save the file.")
        return None
