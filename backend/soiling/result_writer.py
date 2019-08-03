
import uuid
from pathlib import Path
from typing import List
import pandas as pd
from openpyxl.utils.dataframe import dataframe_to_rows

from openpyxl import Workbook


class ResultWriter():

    def __init__(self, workbook_name, worksheet_names):
        self.workbook_name = workbook_name
        self.worksheet_names = worksheet_names

        self.__workbook = Workbook()
        self.__generate_worksheets()

    def __generate_worksheets(self):
        # Replace the first default sheet
        for idx, name in enumerate(self.worksheet_names):
            ws = self.__workbook.create_sheet(name, idx)

    def write_df_to_sheet(self, df: pd.DataFrame, sheet_name: str):
        for r in dataframe_to_rows(df, index=True, header=True):
            self.__workbook[sheet_name].append(r)

    def save_workbook(self, directory_path: str = '/tmp/') -> str:
        """
        Save the workbook to file (workbook_name).

        Arguments:
        directory_path -- Save in supplied directory

        Returns:
        The path of the saved workbook.

        """
        # TODO: Change to date and time instead of UUID
        file_name = self.workbook_name + '-' + str(uuid.uuid4()) + '.xlsx'
        save_path = str(Path(directory_path) / file_name)
        self.__workbook.save(save_path)
        return save_path
