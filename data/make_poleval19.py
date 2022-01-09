"""
Script for praparing raw poleval19.csv file

It concatenates raw files:
- test_set_clean_only_tags.txt
- training_set_clean_only_text.txt
- test_set_clean_only_tags.txt
- test_set_clean_only_text.txt

Enabling loading this dataset with standard reader (CsvDataLoader)
"""
import os
import pandas as pd


if __name__ == '__main__':
    folder = './data/'

    path = os.path.join(folder, 'training_set_clean_only_text.txt')
    with open(path, 'r', encoding="utf8") as f:
        train_text = f.readlines()
        train_text = pd.Series([t[:-1] for t in train_text])

    path = os.path.join(folder, 'training_set_clean_only_tags.txt')
    with open(path, 'r', encoding="utf8") as f:
        train_tags = f.readlines()
        train_tags = pd.Series([t[:-1] for t in train_tags])

    assert len(train_text) == len(train_tags), 'Len of text and tags does not match'

    train_data = pd.DataFrame(zip(train_text, train_tags), columns=['text_raw', 'tag'])
    train_data['dataset'] = 'train'

    path = os.path.join(folder, 'test_set_clean_only_text.txt')
    with open(path, 'r', encoding="utf8") as f:
        test_text = f.readlines()
        test_text = pd.Series([t[:-1] for t in test_text])

    path = os.path.join(folder, 'test_set_clean_only_tags.txt')
    with open(path, 'r', encoding="utf8") as f:
        test_tags = f.readlines()
        test_tags = pd.Series([t[:-1] for t in test_tags])

    assert len(test_text) == len(test_tags), 'Len of text and tags does not match'

    test_data = pd.DataFrame(zip(test_text, test_tags), columns=['text_raw', 'tag'])
    test_data['dataset'] = 'test'

    pol_eval = pd.concat([train_data, test_data], axis=0, ignore_index=True)
    pol_eval = pol_eval[['dataset', 'text_raw', 'tag']]

    pol_eval.to_csv('./data/poleval19.csv', encoding='UTF-8', index=False)
