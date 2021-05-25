import pandas as pd
from pathlib import Path
import numpy as np
from pandas_profiling import ProfileReport


# def load_data(
#         data_fp: str = 'special_projects/entity_matching/data',
#         file_path_or_ext: str = 'xlsx',
#         explore: bool = False,
#         **kwargs
# ) -> tuple:
#     """
#         Loads data file into pandas and extracts the indices of all observations
#     where signal is present, before splitting them into whether the principal
#     entity is mentioned or not.

#     :param data_fp: path to data folder
#     :param file_path_or_ext: file path or extension of data file
#     :param explore: if True, performs data profiling and save html report
#     :param kwargs: additional arguments to pandas .read_xx method
#     :return: a tuple of (full dataset, index of is_entity, index of not_entity)
#     """
#     data_fp = Path(data_fp)

#     if '.' in file_path_or_ext:
#         df = pd.read_excel(
#             data_fp / file_path_or_ext, sheet_name='News Articles')
#     else:
#         news_data = list(data_fp.glob(f'*.{file_path_or_ext}'))[0]
#         df = pd.read_excel(news_data, **kwargs)
#         df = df[[col for col in df.columns if not col.startswith('Unnamed:')]]

def load_data(
        data_fp: str = 'special_projects/entity_matching/data',
        file_path_or_ext: str = 'xlsx',
        explore: bool = False,
        **kwargs
) -> tuple:
    """
        Loads data file into pandas and extracts the indices of all observations
    where signal is present, before splitting them into whether the principal
    entity is mentioned or not.

    :param data_fp: path to data folder
    :param file_path_or_ext: file path or extension of data file
    :param explore: if True, performs data profiling and save html report
    :param kwargs: additional arguments to pandas .read_xx method
    :return: a tuple of (full dataset, index of is_entity, index of not_entity)
    """
    data_fp = Path(data_fp)
    df = pd.read_excel(
    '/content/gdrive/MyDrive/custom-EM-BERT/prof_entity/data/News-Articles-Tagging.xlsx', sheet_name='News Articles')


    # # Explore data
    # if explore:
    #     report = ProfileReport(
    #         df, title='News Articles Profile Report', minimal=True)
    #     report.to_file(data_fp / 'profile_report.html')

    # Slice is_entity and not_entity from all sentences that have a signal
    signal_present = df[
        df.loc[:, '1. revenue':"11. Competitors' fundraising"].sum(axis=1) > 0]
    
    df2 = df[df['entity (WIP)'] == 0]
    signal_present = pd.concat([signal_present, df2])

    is_entity = signal_present[signal_present['entity (WIP)'] == 1].index
    not_entity = signal_present[signal_present['entity (WIP)'] == 0].index

    return df, is_entity, not_entity, signal_present


def create_sample(df: pd.DataFrame, indices: list, n: int = 3) -> list:
    """
        Creates samples from full dataframe using indices and n rows before.
    Returns a list of [joined texts from n rows before, current sentence].

    :param df: full dataframe
    :param indices: list of index numbers of dataframe to extract
    :param n: number of prior rows to return
    :return: a list of [joined texts from n rows before, current sentence]

    """
    samples = []
    for idx in indices:
        if idx <= n:
            continue

        samples.append([
            ' '.join(df.loc[idx - n:idx - 1, 'article'].to_list()),
            df.loc[idx, 'article']
        ])
    return samples
