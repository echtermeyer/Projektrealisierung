import re
import csv
import xmltodict

import pandas as pd

from tqdm import tqdm
from pathlib import Path


class Regex_Extractor:
    @staticmethod
    def classify_entry_row(text):
        pattern = r"\b(\w+):$"
        match = re.search(pattern, text)
        if match:
            return match.group(1).lower()
        return None

    @staticmethod
    def extract_CalculateWeightAndTrimAction(entry_string):
        pattern = re.compile(
            r"([A-Z_]+(?:\s[A-Z_]+)*\s*(?:weight|index)?)\s*:\s*([-\d\.]*)(?:\s*KG)?"
        )
        extracted_dict = {}
        matches = pattern.finditer(entry_string)
        for match in matches:
            key = match.group(1).replace(" ", "_").strip("_")
            value = match.group(2).strip()
            extracted_dict[key] = float(value) if value and value != "NULL" else None
        return extracted_dict

    @staticmethod
    def extract_header_id(log_entry):
        pattern = re.compile(r"\[([a-f0-9]+)\]")
        match = pattern.search(log_entry)

        if match:
            return match.group(1)
        else:
            return None


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


class Preprocessor:
    def __init__(self) -> None:
        self.regex_extractor = Regex_Extractor()

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Preprocessing data...")
        print("Extracting header category and header id...")
        df["header_category"] = df["header_line"].apply(
            self.regex_extractor.classify_entry_row
        )
        df["header_id"] = df["header_line"].apply(
            self.regex_extractor.extract_header_id
        )
        df["creation_time"] = df["creation_time"].apply(pd.to_datetime)
        print("Processing ASMMsgProcessor...")
        df = self.__process_ASMMsgProcessor(df)

        print("Processing CalculateWeightAndTrimAction...")
        df = self.__process_CalculateWeightAndTrimAction(df)

        print("Deriving flight id...")
        df = self.__derive_flight_id(df)

        return df

    def __process_ASMMsgProcessor(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered = df[
            (df["action_name"] == "ASMMsgProcessor")
            & (df["header_category"] == "received")
        ]

        pairs = {}
        for _, row in filtered.iterrows():
            xml_data = row["entry_details"]
            dict_data = xmltodict.parse(xml_data)

            if "newFlight" in dict_data["ns2:OSSChangeMessage"]:
                leg_data = dict_data["ns2:OSSChangeMessage"]["newFlight"]["leg"]
            elif "oldFlight" in dict_data["ns2:OSSChangeMessage"]:
                leg_data = dict_data["ns2:OSSChangeMessage"]["oldFlight"]["leg"]
            else:
                leg_data = {}

            if isinstance(leg_data, list):
                leg_data = leg_data[-1]

            pairs[row["header_id"]] = leg_data

        pairs_df = pd.DataFrame.from_dict(pairs, orient="index")
        pairs_df.reset_index(inplace=True)
        pairs_df.rename(columns={"index": "header_id"}, inplace=True)

        if "id" in pairs_df.columns:
            pairs_df.drop("id", axis=1, inplace=True)

        return pd.merge(df, pairs_df, on="header_id", how="left")

    def __process_CalculateWeightAndTrimAction(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered = df[
            (df["action_name"] == "CalculateWeightAndTrimAction")
            & (df["header_category"] == "saved")
        ]

        pairs = {}
        for _, row in filtered.iterrows():
            raw_data = row["entry_details"]
            extracted_data = self.regex_extractor.extract_CalculateWeightAndTrimAction(
                raw_data
            )

            pairs[row["id"]] = extracted_data

        pairs_df = pd.DataFrame.from_dict(pairs, orient="index")
        pairs_df.reset_index(inplace=True)
        pairs_df.rename(columns={"index": "id"}, inplace=True)

        return pd.merge(df, pairs_df, on="id", how="left")

    def __derive_flight_id(self, df: pd.DataFrame) -> pd.DataFrame:
        df['flight_id'] = (
            df['airline_code'].astype(str) + '_' +
            df['flight_number'].astype(str) + '_' +
            df['flight_date'].astype(str) + '_' +
            df['departure_airport'].astype(str)
        )

        df = df[["flight_id"] + [col for col in df.columns if col != "flight_id"]]
        
        return df
