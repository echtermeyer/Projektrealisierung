import csv
import pandas as pd

from tqdm import tqdm
from pathlib import Path


class TripLoader:
    def __init__(self, fix_csv_errors: bool = True) -> None:
        self.csv_cleaner = CSV_Cleaner()
        self.fix_csv_errors = fix_csv_errors

        self.ROOT = Path(__file__).parent.parent
        self.file_paths = {
            "trips": "src/data/tripfile.csv",
            "trips_ABCD": "src/data/ABCD_tripfiles.csv",
            "trips_MNOP": "src/data/MNOP_tripfiles.csv",
            "trips_ZYXW": "src/data/ZYXW_tripfiles.csv",
        }

    def __load(self, dataset: str, delimiter: str = ",") -> pd.DataFrame:
        return pd.read_csv(self.ROOT / self.file_paths[dataset], delimiter=delimiter)

    @property
    def trips(self) -> pd.DataFrame:
        if hasattr(self, "_trips_data"):
            return self._trips_data
        self._trips_data = self.__load("trips", delimiter=";")
        return self._trips_data

    @property
    def trips_ABCD(self) -> pd.DataFrame:
        if hasattr(self, "_trips_ABCD_data"):
            return self._trips_ABCD_data

        if self.fix_csv_errors:
            self.csv_cleaner.fix_csv(self.ROOT / self.file_paths["trips_ABCD"])

        self._trips_ABCD_data = self.__load("trips_ABCD")
        return self._trips_ABCD_data

    @property
    def trips_MNOP(self) -> pd.DataFrame:
        if hasattr(self, "_trips_MNOP_data"):
            return self._trips_MNOP_data

        if self.fix_csv_errors:
            self.csv_cleaner.fix_csv(self.ROOT / self.file_paths["trips_MNOP"])

        self._trips_MNOP_data = self.__load("trips_MNOP")
        return self._trips_MNOP_data

    @property
    def trips_ZYXW(self) -> pd.DataFrame:
        if hasattr(self, "_trips_ZYXW_data"):
            return self._trips_ZYXW_data

        if self.fix_csv_errors:
            self.csv_cleaner.fix_csv(self.ROOT / self.file_paths["trips_ZYXW"])

        self._trips_ZYXW_data = self.__load("trips_ZYXW")
        return self._trips_ZYXW_data


class CSV_Cleaner:
    def __analyse_csv(self, file_path):
        incorrect_ids = []
        with open(file_path, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            columns = next(reader)
            for row in tqdm(reader, desc="Processing rows"):
                if len(row) > len(columns):
                    incorrect_ids.append(row[0])

        return incorrect_ids

    def fix_csv(self, file_path: Path):
        incorrect_ids = self.__analyse_csv(file_path)

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        for row_id in incorrect_ids:
            start_index = content.find(row_id)
            if start_index == -1:
                continue

            # Find the 11th comma after the start index
            comma_count = 0
            current_index = start_index
            while comma_count < 11 and current_index < len(content):
                if content[current_index] == ",":
                    comma_count += 1
                current_index += 1

            while current_index >= 0 and content[current_index] != "\n":
                current_index -= 1

            content = content[:current_index] + '"' + content[current_index:]

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
