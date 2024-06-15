import re
import csv
import xmltodict

import pandas as pd

from tqdm import tqdm
from pathlib import Path

from typing import Dict, List

from src.utils import (
    COLUMNS_CalculateWeightAndTrimAction,
    COLUMNS_AssignLCCAction,
    COLUMNS_UpdateFlightAction_METADATA,
    COLUMNS_UpdateFlightAction_RECEIVED,
    COLUMNS_UpdateFlightAction_SAVED,
    COLUMNS_UpdateCrewDataAction,
    COLUMNS_StoreRegistrationAndConfigurationAc,
    COLUMNS_StoreRegistrationAndConfigurationAc_STATUS_KEYS,
    COLUMNS_UpdateLoadTableAction,
    COLUMNS_UpdateLoadTableAction_STATUS_KEYS,
    COLUMNS_StorePaxDataAction_saved,
    COLUMNS_StorePaxDataAction_STATUS_KEYS_saved,
)


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
    def extract_AssignLCCAction(entry_string):
        data = entry_string.split("\n")[1]

        values = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}|\S+", data)
        return dict(zip(COLUMNS_AssignLCCAction, values))

    @staticmethod
    def extract_UpdateFlightAction(entry_string: str, header_category: str):
        if header_category == "received":
            leg_keys = COLUMNS_UpdateFlightAction_RECEIVED
        elif header_category == "saved":
            leg_keys = COLUMNS_UpdateFlightAction_SAVED

        lines = entry_string.split("\n")

        extracted_dict = {}
        for line in lines[0:1]:
            for key in COLUMNS_UpdateFlightAction_METADATA:
                pattern = f"{key}: (.*?)(?=\s+\w+:|$)"
                match = re.search(pattern, line)
                if match:
                    extracted_dict[key] = match.group(1).strip()

        legs = []
        legs_start_index = lines.index("Legs:") + 2
        for line in lines[legs_start_index:]:
            if line.strip() == "":
                continue

            values = re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}|\S+", line)
            leg_data = dict(zip(leg_keys, values))
            legs.append(leg_data)

        extracted_dict["legs"] = legs
        return extracted_dict

    @staticmethod
    def extract_UpdateCrewDataAction(entry_string: str):
        combined_lines = " ".join(entry_string.split("\n")[1:])

        pattern = "|".join(re.escape(key) for key in COLUMNS_UpdateCrewDataAction)
        parts = re.split(f"({pattern})", combined_lines)

        key = None
        extracted_dict = {}
        for part in parts:
            if part in COLUMNS_UpdateCrewDataAction:
                key = part
            elif key:
                value = part.lstrip(":").strip().lstrip(":").strip()

                if value == "NULL":
                    value = None
                elif value.isdigit():
                    value = int(value)

                extracted_dict[key] = value
                key = None

        return extracted_dict

    @staticmethod
    def extract_StoreRegistrationAndConfigurationAc(entry_string: str):
        extracted_dict = {}
        for key in COLUMNS_StoreRegistrationAndConfigurationAc:
            pattern = rf"{re.escape(key)}\s*:\s*([\d./]+|NULL)"
            match = re.search(pattern, entry_string)
            if match:
                value = match.group(1)
                if value == "NULL":
                    extracted_dict[key] = None
                elif value.replace(".", "", 1).isdigit():
                    extracted_dict[key] = float(value)
                else:
                    extracted_dict[key] = value

        status_data = {}
        status_key_pattern = re.compile(r"STATUS\s+(.*)")
        status_part = status_key_pattern.search(entry_string)
        if status_part:
            statuses = status_part.group(1).split()
            for i in range(0, len(statuses), 2):
                key = "STATUS_" + statuses[i]
                status_data[key] = int(statuses[i + 1])

        for key in COLUMNS_StoreRegistrationAndConfigurationAc_STATUS_KEYS:
            prefixed_key = "STATUS_" + key
            if prefixed_key not in status_data:
                status_data[prefixed_key] = None

        extracted_dict.update(status_data)
        return extracted_dict

    @staticmethod
    def extract_UpdateLoadTableAction(entry_string: str):
        extracted_dict = {}

        for key in COLUMNS_UpdateLoadTableAction:
            pattern = rf"{re.escape(key)}\s*:\s*([\d.]+)"
            match = re.search(pattern, entry_string)
            if match:
                extracted_dict[f"ESTIMATED_{key.replace(' ', '_')}"] = float(
                    match.group(1)
                )

        status_data = {}
        status_key_pattern = re.compile(r"STATUS\s+(.*)")
        status_part = status_key_pattern.search(entry_string)
        if status_part:
            statuses = status_part.group(1).split()
            for i in range(0, len(statuses), 2):
                key = "STATUS_" + statuses[i]
                status_data[key] = int(statuses[i + 1])

        for key in COLUMNS_UpdateLoadTableAction_STATUS_KEYS:
            prefixed_key = "STATUS_" + key
            if prefixed_key not in status_data:
                status_data[prefixed_key] = None

        extracted_dict.update(status_data)
        return extracted_dict

    @staticmethod
    def extract_StorePaxDataAction_saved(entry_string: str):
        config_data = {}

        for key in COLUMNS_StorePaxDataAction_saved:
            if key in ["Total bag weight"]:  # Extracting numeric values with units
                pattern = rf"{re.escape(key)}\s*:\s*([\d.]+)\s*KG"  # Extracting just the number before "KG"
            elif key in [
                "Baggage weight type"
            ]:  # Extracting strings that could include spaces or special characters
                pattern = rf"{re.escape(key)}\s*:\s*([a-zA-Z_]+)"  # Adjusted to capture proper string formats
            else:
                pattern = rf"{re.escape(key)}\s*:\s*([\d]+|NULL)"  # For numeric or NULL values

            match = re.search(pattern, entry_string)
            if match:
                value = match.group(1)
                if value.isdigit():
                    config_data[key] = int(value)
                elif value == "NULL":
                    config_data[key] = None
                else:
                    try:
                        config_data[key] = float(value)
                    except ValueError:
                        config_data[key] = value

        distribution_match = re.search(r"Distribution\s*:\s*([a-zA-Z_]+)", entry_string)
        if distribution_match:
            config_data["Distribution"] = distribution_match.group(1)

        status_data = {}
        status_key_pattern = re.compile(r"STATUS\s+(.*)")
        status_part = status_key_pattern.search(entry_string)
        if status_part:
            statuses = status_part.group(1).split()
            for i in range(0, len(statuses), 2):
                key = "STATUS_" + statuses[i]
                status_data[key] = int(statuses[i + 1])

        for key in COLUMNS_StorePaxDataAction_STATUS_KEYS_saved:
            prefixed_key = "STATUS_" + key
            if prefixed_key not in status_data:
                status_data[prefixed_key] = None

        config_data.update(status_data)
        return config_data

    @staticmethod
    def extract_StorePaxDataAction_received(input_text: str):
        baggage_weight_type_pattern = r"Baggage weight type:\s*(\S+)"
        distribution_pattern = r"Distribution\s*:\s*(\S+)"

        baggage_weight_type = re.search(baggage_weight_type_pattern, input_text).group(
            1
        )
        distribution = re.search(distribution_pattern, input_text).group(1)

        def extract_labels(input_text):
            pattern = r"(\S+)\s{2,}"
            for line in input_text.split("\n"):
                if re.search(pattern, line):
                    labels = re.findall(pattern, line)
                    if len(labels) > 5:
                        return labels

        headers = extract_labels(input_text)

        pax_type_pattern = r"(Checkin|Loadsheet)\s+(.*)"
        pax_type_search = re.search(pax_type_pattern, input_text)
        pax_type = pax_type_search.group(1)
        pax_values_line = pax_type_search.group(2)
        pax_values_line = re.sub(r"\s+KG", "KG", pax_values_line)
        pax_values = pax_values_line.split()

        def parse_value(value):
            if value == "NULL":
                return None
            value = value.replace("KG", "").strip()
            if value == "":
                return None
            return float(value)

        values = {key: parse_value(val) for key, val in zip(headers, pax_values)}

        data = {
            "Baggage weight type": baggage_weight_type,
            "Distribution": distribution,
            "Pax type": pax_type,
        }
        data.update(values)

        return data

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


class DF_Cleaner:
    @staticmethod
    def remove_column_anonymization(
        df: pd.DataFrame,
        column_dict: Dict[str, str],
    ) -> pd.DataFrame:
        return df.rename(columns=column_dict)


class Preprocessor:
    def __init__(self, root: Path) -> None:
        self.DATA_DIR = root / "src/data/extracted/"
        self.DATA_DIR.mkdir(exist_ok=True)

        self.df_cleaner = DF_Cleaner()
        self.regex_extractor = Regex_Extractor()

    def preprocess(self, df: pd.DataFrame, dataset: str) -> pd.DataFrame:
        print("Preprocessing data...")

        print("-- Inserting column for extracted data...")
        df["extracted_data_path"] = None
        prefix = dataset.split("_")[1].lower()

        print("-- Extracting header category and header id...")
        df["header_category"] = df["header_line"].apply(
            self.regex_extractor.classify_entry_row
        )
        df["header_id"] = df["header_line"].apply(
            self.regex_extractor.extract_header_id
        )
        df["creation_time"] = df["creation_time"].apply(pd.to_datetime)

        print("-- Processing ASMMsgProcessor...")
        df = self.__process_ASMMsgProcessor(df)

        print("-- Deriving flight id...")
        df = self.__derive_flight_id(df)

        print("-- Processing CalculateWeightAndTrimAction...")
        df = self.__process_CalculateWeightAndTrimAction(df, prefix)

        print("-- Processing AssignLCCAction...")
        df = self.__process_AssignLCCAction(df, prefix)

        print("-- Processing UpdateFlightAction...")
        df = self.__process_UpdateFlightAction(df, prefix)

        print("-- Processing UpdateCrewDataAction...")
        df = self.__process_UpdateCrewDataAction(df, prefix)

        print("-- Processing StoreRegistrationAndConfigurationAc...")
        df = self.__process_StoreRegistrationAndConfigurationAc(df, prefix)

        print("-- Processing UpdateLoadTableAction...")
        df = self.__process_UpdateLoadTableAction(df, prefix)

        print("-- Processing StorePaxDataAction...")
        df = self.__process_StorePaxDataAction_saved(df, prefix)
        df = self.__process_StorePaxDataAction_received(df, prefix)

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

    def __process_CalculateWeightAndTrimAction(
        self, df: pd.DataFrame, prefix: str
    ) -> pd.DataFrame:
        NAME = "CalculateWeightAndTrimAction"
        PATH = self.DATA_DIR / f"{prefix}_{NAME}.csv"

        filtered = df[(df["action_name"] == NAME) & (df["header_category"] == "saved")]

        data = {}
        for _, row in filtered.iterrows():
            raw_data = row["entry_details"]
            extracted_data = self.regex_extractor.extract_CalculateWeightAndTrimAction(
                raw_data
            )
            extracted_data["flight_id"] = row["flight_id"]
            extracted_data["action_name"] = row["action_name"]

            data[row["id"]] = extracted_data

        data_df = pd.DataFrame.from_dict(data, orient="index")
        data_df.reset_index(inplace=True)
        data_df.rename(columns={"index": "id"}, inplace=True)

        first_columns = ["flight_id", "id", "action_name"]
        following_columns = [col for col in data_df.columns if col not in first_columns]

        data_df = data_df[first_columns + following_columns]
        data_df = self.df_cleaner.remove_column_anonymization(
            data_df, COLUMNS_CalculateWeightAndTrimAction
        )
        data_df.to_csv(PATH, index=False)

        df.loc[df["id"].isin(data_df["id"]), "extracted_data"] = PATH
        return df

    def __process_AssignLCCAction(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        NAME = "AssignLCCAction"
        PATH = self.DATA_DIR / f"{prefix}_{NAME}.csv"

        filtered = df[(df["action_name"] == NAME) & (df["header_category"] == "saved")]

        data = {}
        for _, row in filtered.iterrows():
            raw_data = row["entry_details"]
            extracted_data = self.regex_extractor.extract_AssignLCCAction(raw_data)
            extracted_data["flight_id"] = row["flight_id"]
            extracted_data["action_name"] = row["action_name"]

            data[row["id"]] = extracted_data

        data_df = pd.DataFrame.from_dict(data, orient="index")
        data_df.reset_index(inplace=True)
        data_df.rename(columns={"index": "id"}, inplace=True)

        first_columns = ["flight_id", "id", "action_name"]
        following_columns = [col for col in data_df.columns if col not in first_columns]

        data_df = data_df[first_columns + following_columns]
        data_df = self.df_cleaner.remove_column_anonymization(
            data_df, COLUMNS_CalculateWeightAndTrimAction
        )
        data_df.to_csv(PATH, index=False)

        df.loc[df["id"].isin(data_df["id"]), "extracted_data"] = PATH
        return df

    def __process_UpdateFlightAction(
        self, df: pd.DataFrame, prefix: str
    ) -> pd.DataFrame:
        NAME = "UpdateFlightAction"

        for header_category in ["received", "saved"]:
            PATH = self.DATA_DIR / f"{prefix}_{NAME}_{header_category}.csv"

            filtered = df[
                (df["action_name"] == NAME) & (df["header_category"] == header_category)
            ]

            data = {}
            for _, row in filtered.iterrows():
                raw_data = row["entry_details"]
                extracted_data = self.regex_extractor.extract_UpdateFlightAction(
                    raw_data, header_category
                )
                extracted_data["flight_id"] = row["flight_id"]
                extracted_data["action_name"] = row["action_name"]

                data[row["id"]] = extracted_data

            data_df = pd.DataFrame.from_dict(data, orient="index")
            data_df.reset_index(inplace=True)
            data_df.rename(columns={"index": "id"}, inplace=True)

            first_columns = ["flight_id", "id", "action_name"]
            following_columns = [
                col for col in data_df.columns if col not in first_columns
            ]

            data_df = data_df[first_columns + following_columns]
            data_df = self.df_cleaner.remove_column_anonymization(
                data_df, COLUMNS_CalculateWeightAndTrimAction
            )
            data_df.to_csv(PATH, index=False)

            df.loc[df["id"].isin(data_df["id"]), "extracted_data"] = PATH

        return df

    def __process_UpdateCrewDataAction(
        self, df: pd.DataFrame, prefix: str
    ) -> pd.DataFrame:
        NAME = "UpdateCrewDataAction"
        PATH = self.DATA_DIR / f"{prefix}_{NAME}.csv"

        filtered = df[
            (df["action_name"] == NAME) & (df["header_category"] == "received")
        ]

        data = {}
        for _, row in filtered.iterrows():
            raw_data = row["entry_details"]
            extracted_data = self.regex_extractor.extract_UpdateCrewDataAction(raw_data)
            extracted_data["flight_id"] = row["flight_id"]
            extracted_data["action_name"] = row["action_name"]

            data[row["id"]] = extracted_data

        data_df = pd.DataFrame.from_dict(data, orient="index")
        data_df.reset_index(inplace=True)
        data_df.rename(columns={"index": "id"}, inplace=True)

        first_columns = ["flight_id", "id", "action_name"]
        following_columns = [col for col in data_df.columns if col not in first_columns]

        data_df = data_df[first_columns + following_columns]
        data_df = self.df_cleaner.remove_column_anonymization(
            data_df, COLUMNS_CalculateWeightAndTrimAction
        )
        data_df.to_csv(PATH, index=False)

        df.loc[df["id"].isin(data_df["id"]), "extracted_data"] = PATH
        return df

    def __process_StoreRegistrationAndConfigurationAc(
        self, df: pd.DataFrame, prefix: str
    ) -> pd.DataFrame:
        NAME = "StoreRegistrationAndConfigurationAc"
        PATH = self.DATA_DIR / f"{prefix}_{NAME}.csv"

        filtered = df[(df["action_name"] == NAME) & (df["header_category"] == "saved")]

        data = {}
        for _, row in filtered.iterrows():
            raw_data = row["entry_details"]
            extracted_data = (
                self.regex_extractor.extract_StoreRegistrationAndConfigurationAc(
                    raw_data
                )
            )
            extracted_data["flight_id"] = row["flight_id"]
            extracted_data["action_name"] = row["action_name"]

            data[row["id"]] = extracted_data

        data_df = pd.DataFrame.from_dict(data, orient="index")
        data_df.reset_index(inplace=True)
        data_df.rename(columns={"index": "id"}, inplace=True)

        first_columns = ["flight_id", "id", "action_name"]
        following_columns = [col for col in data_df.columns if col not in first_columns]

        data_df = data_df[first_columns + following_columns]
        data_df = self.df_cleaner.remove_column_anonymization(
            data_df, COLUMNS_CalculateWeightAndTrimAction
        )
        data_df.to_csv(PATH, index=False)

        df.loc[df["id"].isin(data_df["id"]), "extracted_data"] = PATH
        return df

    def __process_UpdateLoadTableAction(
        self, df: pd.DataFrame, prefix: str
    ) -> pd.DataFrame:
        NAME = "UpdateLoadTableAction"
        PATH = self.DATA_DIR / f"{prefix}_{NAME}.csv"

        filtered = df[(df["action_name"] == NAME) & (df["header_category"] == "saved")]

        data = {}
        for _, row in filtered.iterrows():
            raw_data = row["entry_details"]
            extracted_data = self.regex_extractor.extract_UpdateLoadTableAction(
                raw_data
            )
            extracted_data["flight_id"] = row["flight_id"]
            extracted_data["action_name"] = row["action_name"]

            data[row["id"]] = extracted_data

        data_df = pd.DataFrame.from_dict(data, orient="index")
        data_df.reset_index(inplace=True)
        data_df.rename(columns={"index": "id"}, inplace=True)

        first_columns = ["flight_id", "id", "action_name"]
        following_columns = [col for col in data_df.columns if col not in first_columns]

        data_df = data_df[first_columns + following_columns]
        data_df = self.df_cleaner.remove_column_anonymization(
            data_df, COLUMNS_CalculateWeightAndTrimAction
        )
        data_df.to_csv(PATH, index=False)

        df.loc[df["id"].isin(data_df["id"]), "extracted_data"] = PATH
        return df

    def __process_StorePaxDataAction_saved(
        self, df: pd.DataFrame, prefix: str
    ) -> pd.DataFrame:
        NAME = "StorePaxDataAction"
        PATH = self.DATA_DIR / f"{prefix}_{NAME}_saved.csv"

        filtered = df[(df["action_name"] == NAME) & (df["header_category"] == "saved")]

        data = {}
        for _, row in filtered.iterrows():
            raw_data = row["entry_details"]
            extracted_data = self.regex_extractor.extract_StorePaxDataAction_saved(
                raw_data
            )
            extracted_data["flight_id"] = row["flight_id"]
            extracted_data["action_name"] = row["action_name"]

            data[row["id"]] = extracted_data

        data_df = pd.DataFrame.from_dict(data, orient="index")
        data_df.reset_index(inplace=True)
        data_df.rename(columns={"index": "id"}, inplace=True)

        first_columns = ["flight_id", "id", "action_name"]
        following_columns = [col for col in data_df.columns if col not in first_columns]

        data_df = data_df[first_columns + following_columns]
        data_df = self.df_cleaner.remove_column_anonymization(
            data_df, COLUMNS_CalculateWeightAndTrimAction
        )
        data_df.to_csv(PATH, index=False)

        df.loc[df["id"].isin(data_df["id"]), "extracted_data"] = PATH
        return df

    def __process_StorePaxDataAction_received(
        self, df: pd.DataFrame, prefix: str
    ) -> pd.DataFrame:
        NAME = "StorePaxDataAction"
        PATH = self.DATA_DIR / f"{prefix}_{NAME}_saved.csv"

        filtered = df[
            (df["action_name"] == NAME) & (df["header_category"] == "received")
        ]

        data = {}
        for _, row in filtered.iterrows():
            raw_data = row["entry_details"]
            extracted_data = self.regex_extractor.extract_StorePaxDataAction_received(
                raw_data
            )
            extracted_data["flight_id"] = row["flight_id"]
            extracted_data["action_name"] = row["action_name"]

            data[row["id"]] = extracted_data

        data_df = pd.DataFrame.from_dict(data, orient="index")
        data_df.reset_index(inplace=True)
        data_df.rename(columns={"index": "id"}, inplace=True)

        first_columns = ["flight_id", "id", "action_name"]
        following_columns = [col for col in data_df.columns if col not in first_columns]

        data_df = data_df[first_columns + following_columns]
        data_df = self.df_cleaner.remove_column_anonymization(
            data_df, COLUMNS_CalculateWeightAndTrimAction
        )
        data_df.to_csv(PATH, index=False)

        df.loc[df["id"].isin(data_df["id"]), "extracted_data"] = PATH
        return df

    def __derive_flight_id(self, df: pd.DataFrame) -> pd.DataFrame:
        df["flight_id"] = (
            df["airline_code"].astype(str)
            + "_"
            + df["flight_number"].astype(str)
            + "_"
            + df["flight_date"].astype(str)
            + "_"
            + df["departure_airport"].astype(str)
        )

        return df[["flight_id"] + [col for col in df.columns if col != "flight_id"]]
