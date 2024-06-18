import re
import csv
import xmltodict

import pandas as pd

from tqdm import tqdm
from pathlib import Path

from typing import Callable, Dict, List, Optional

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
    COLUMNS_FuelDataInitializer,
    COLUMNS_FuelDataInitializer_STATUS_KEYS,
    COLUMNS_UpdateFuelDataAction_STATUS_KEYS,
    COLUMNS_UpdateFuelDataAction_MW_KEYS,
    COLUMNS_UpdateFuelDataAction_FUEL_KEYS,
    COLUMNS_UpdateFuelDataAction_received,
    COLUMNS_UpdateFuelDataAction_ANONYMIZATION,
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

        entry_string = entry_string.replace("\r", "")
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

        config_data = {
            "Baggage weight type": baggage_weight_type,
            "Distribution": distribution,
            "Pax type": pax_type,
        }
        config_data.update(values)

        return config_data

    @staticmethod
    def extract_FuelDataInitializer(input_text: str):
        status_data = {}
        fuel_data = {}

        lines = input_text.strip().split("\n")

        status_line = lines[0]
        status_parts = status_line.split()
        for idx, part in enumerate(status_parts):
            if part in COLUMNS_FuelDataInitializer_STATUS_KEYS:
                status_data[f"STATUS_{part}"] = int(status_parts[idx + 1])

        fuel_line = next((line for line in lines if line.startswith("FUEL")), None)
        if fuel_line:
            fuel_parts = fuel_line.split()
            for part in fuel_parts:
                key_value = part.split("=")
                if len(key_value) == 2:
                    key, value = key_value
                    if value == "null":
                        fuel_data[f"FUEL_{key}"] = None
                    else:
                        fuel_data[f"FUEL_{key}"] = float(value)
        else:
            for key in COLUMNS_FuelDataInitializer:
                fuel_data[f"FUEL_{key}"] = None

        config_data = {**status_data, **fuel_data}
        return config_data

    @staticmethod
    def extract_UpdateFuelDataAction_saved(input_text: str):
        lines = input_text.split("\n")

        mw_line = next((line for line in lines if line.strip().startswith("MW")), None)
        mw_data = {key: None for key in COLUMNS_UpdateFuelDataAction_MW_KEYS}
        if mw_line:
            pairs = mw_line.split()
            for pair in pairs[1:]:
                if "=" in pair:
                    key, value = pair.split("=")
                    if "KG" in value:
                        value = float(value.replace("KG", ""))
                    else:
                        value = float(value)
                    mw_data[key] = value

        fuel_line = next(
            (line for line in lines if line.strip().startswith("FUEL")), None
        )
        fuel_data = {key: None for key in COLUMNS_UpdateFuelDataAction_FUEL_KEYS}
        if fuel_line:
            pairs = fuel_line.split()
            for pair in pairs[1:]:
                if "=" in pair:
                    key, value = pair.split("=")
                    if value == "null":
                        fuel_data[key] = None
                    elif "KG" in value:
                        fuel_data[key] = float(value.replace("KG", ""))
                    else:
                        fuel_data[key] = float(value)

        status_line = next(
            (line for line in lines if line.strip().startswith("STATUS")), None
        )
        status_data = {
            f"STATUS_{key}": None for key in COLUMNS_UpdateFuelDataAction_STATUS_KEYS
        }
        if status_line:
            parts = status_line.split()
            for idx, part in enumerate(parts):
                if part in COLUMNS_UpdateFuelDataAction_STATUS_KEYS:
                    status_data[f"STATUS_{part}"] = int(parts[idx + 1])

        config_data = {**mw_data, **fuel_data, **status_data}
        return config_data

    @staticmethod
    def extract_UpdateFuelDataAction_received(input_text: str):
        extracted_dict = {}
        for i in range(len(COLUMNS_UpdateFuelDataAction_received)):
            current_key = COLUMNS_UpdateFuelDataAction_received[i]
            if i + 1 < len(COLUMNS_UpdateFuelDataAction_received):
                next_key = COLUMNS_UpdateFuelDataAction_received[i + 1]
                pattern = rf"{current_key}\s*:\s*([^:]*)(?=\s*{next_key}\s*:)"
            else:
                pattern = rf"{current_key}\s*:\s*(.*)"

            match = re.search(pattern, input_text, re.DOTALL)
            if match:
                value = match.group(1).strip()
                if "KG" in value:
                    value = re.sub(r" KGPERCUBICMETER| KG", "", value).strip()
                if value.isdigit() or value.replace(".", "", 1).isdigit():
                    extracted_dict[current_key] = float(value)
                elif value.lower() == "null":
                    extracted_dict[current_key] = None
                elif value.lower() in ["true", "false"]:
                    extracted_dict[current_key] = value.lower() == "true"
                else:
                    extracted_dict[current_key] = value
            else:
                extracted_dict[current_key] = None

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
        steps = [
            (
                "Inserting column for extracted data",
                lambda df: df.assign(extracted_data_path=None),
            ),
            ("Extracting header category and header id", self.__extract_header_info),
            ("Processing ASMMsgProcessor", self.__process_ASMMsgProcessor),
            ("Deriving flight id", self.__derive_flight_id),
            (
                "Processing CalculateWeightAndTrimAction",
                lambda df: self.__process_wrapper(
                    df=df,
                    dataset_name=dataset.split("_")[1].lower(),
                    action_name="CalculateWeightAndTrimAction",
                    header_category="saved",
                    use_suffix=False,
                    fn_extractor=self.regex_extractor.extract_CalculateWeightAndTrimAction,
                    column_anonymization=COLUMNS_CalculateWeightAndTrimAction,
                ),
            ),
            (
                "Processing AssignLCCAction",
                lambda df: self.__process_wrapper(
                    df=df,
                    dataset_name=dataset.split("_")[1].lower(),
                    action_name="AssignLCCAction",
                    header_category="saved",
                    use_suffix=False,
                    fn_extractor=self.regex_extractor.extract_AssignLCCAction,
                ),
            ),
            (
                "Processing UpdateFlightAction",
                lambda df: self.__process_wrapper(
                    df=df,
                    dataset_name=dataset.split("_")[1].lower(),
                    action_name="UpdateFlightAction",
                    header_category="received",
                    use_suffix=True,
                    fn_extractor=self.regex_extractor.extract_UpdateFlightAction,
                    parse_header_category=True,
                ),
            ),
            (
                "Processing UpdateFlightAction (saved)",
                lambda df: self.__process_wrapper(
                    df=df,
                    dataset_name=dataset.split("_")[1].lower(),
                    action_name="UpdateFlightAction",
                    header_category="saved",
                    use_suffix=True,
                    fn_extractor=self.regex_extractor.extract_UpdateFlightAction,
                    parse_header_category=True,
                ),
            ),
            (
                "Processing UpdateCrewDataAction",
                lambda df: self.__process_wrapper(
                    df=df,
                    dataset_name=dataset.split("_")[1].lower(),
                    action_name="UpdateCrewDataAction",
                    header_category="received",
                    use_suffix=False,
                    fn_extractor=self.regex_extractor.extract_UpdateCrewDataAction,
                ),
            ),
            (
                "Processing StoreRegistrationAndConfigurationAc",
                lambda df: self.__process_wrapper(
                    df=df,
                    dataset_name=dataset.split("_")[1].lower(),
                    action_name="StoreRegistrationAndConfigurationAc",
                    header_category="saved",
                    use_suffix=False,
                    fn_extractor=self.regex_extractor.extract_StoreRegistrationAndConfigurationAc,
                ),
            ),
            (
                "Processing UpdateLoadTableAction",
                lambda df: self.__process_wrapper(
                    df=df,
                    dataset_name=dataset.split("_")[1].lower(),
                    action_name="UpdateLoadTableAction",
                    header_category="saved",
                    use_suffix=True,
                    fn_extractor=self.regex_extractor.extract_UpdateLoadTableAction,
                ),
            ),
            (
                "Processing StorePaxDataAction (saved)",
                lambda df: self.__process_wrapper(
                    df=df,
                    dataset_name=dataset.split("_")[1].lower(),
                    action_name="StorePaxDataAction",
                    header_category="saved",
                    use_suffix=True,
                    fn_extractor=self.regex_extractor.extract_StorePaxDataAction_saved,
                ),
            ),
            (
                "Processing StorePaxDataAction (received)",
                lambda df: self.__process_wrapper(
                    df=df,
                    dataset_name=dataset.split("_")[1].lower(),
                    action_name="StorePaxDataAction",
                    header_category="received",
                    use_suffix=True,
                    fn_extractor=self.regex_extractor.extract_StorePaxDataAction_received,
                ),
            ),
            (
                "Processing FuelDataInitializer",
                lambda df: self.__process_wrapper(
                    df=df,
                    dataset_name=dataset.split("_")[1].lower(),
                    action_name="FuelDataInitializer",
                    header_category="saved",
                    use_suffix=False,
                    fn_extractor=self.regex_extractor.extract_FuelDataInitializer,
                ),
            ),
            (
                "Processing UpdateFuelDataAction (saved)",
                lambda df: self.__process_wrapper(
                    df=df,
                    dataset_name=dataset.split("_")[1].lower(),
                    action_name="UpdateFuelDataAction",
                    header_category="saved",
                    use_suffix=True,
                    fn_extractor=self.regex_extractor.extract_UpdateFuelDataAction_saved,
                    column_anonymization=COLUMNS_UpdateFuelDataAction_ANONYMIZATION,
                ),
            ),
            (
                "Processing UpdateFuelDataAction (received)",
                lambda df: self.__process_wrapper(
                    df=df,
                    dataset_name=dataset.split("_")[1].lower(),
                    action_name="UpdateFuelDataAction",
                    header_category="received",
                    use_suffix=True,
                    fn_extractor=self.regex_extractor.extract_UpdateFuelDataAction_received,
                ),
            ),
        ]

        with tqdm(total=len(steps), desc="Preprocessing Steps") as pbar:
            for description, function in steps:
                pbar.set_description(description)
                df = function(df)
                pbar.update(1)

        return df

    def __extract_header_info(self, df):
        df["header_category"] = df["header_line"].apply(
            self.regex_extractor.classify_entry_row
        )
        df["header_id"] = df["header_line"].apply(
            self.regex_extractor.extract_header_id
        )
        df["creation_time"] = df["creation_time"].apply(pd.to_datetime)
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

    def __process_wrapper(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        action_name: str,
        header_category: str,
        use_suffix: bool,
        fn_extractor: Callable,
        column_anonymization: Optional[List[str]] = None,
        parse_header_category: bool = False,
    ) -> pd.DataFrame:
        if use_suffix:
            PATH = self.DATA_DIR / f"{dataset_name}_{action_name}_{header_category}.csv"
        else:
            PATH = self.DATA_DIR / f"{dataset_name}_{action_name}.csv"

        filtered = df[
            (df["action_name"] == action_name)
            & (df["header_category"] == header_category)
        ]

        data = {}
        for _, row in filtered.iterrows():
            raw_data = row["entry_details"]
            if parse_header_category:
                extracted_data = fn_extractor(raw_data, header_category)
            else:
                extracted_data = fn_extractor(raw_data)
            extracted_data["flight_id"] = row["flight_id"]
            extracted_data["action_name"] = row["action_name"]

            data[row["id"]] = extracted_data

        data_df = pd.DataFrame.from_dict(data, orient="index")
        data_df.reset_index(inplace=True)
        data_df.rename(columns={"index": "id"}, inplace=True)

        first_columns = ["flight_id", "id", "action_name"]
        following_columns = [col for col in data_df.columns if col not in first_columns]

        data_df = data_df[first_columns + following_columns]
        if column_anonymization:
            data_df = self.df_cleaner.remove_column_anonymization(
                data_df, column_anonymization
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
