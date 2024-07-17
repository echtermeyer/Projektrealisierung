import csv
import re

input_file_path = "airports.txt"
output_file_path = "airports.csv"


# Funktion zur Extraktion und Umwandlung der Daten in CSV
def convert_to_csv(input_path, output_path):
    # Daten lesen und Zeilen verarbeiten
    with open(input_path, "r") as file:
        lines = file.readlines()

    data = []

    # Regex zum Extrahieren der Informationen aus jeder Zeile
    pattern = re.compile(r"'(\w{3})': \[([-.\d]+), ([-.\d]+)\],\s*#\s*(.*),\s*(.*)")

    # Zeilen parsen und relevante Informationen extrahieren
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            airport_code = match.group(1)
            latitude = match.group(2)
            longitude = match.group(3)
            city = match.group(4)
            country = match.group(5)
            data.append([airport_code, latitude, longitude, city, country])

    with open(output_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Schreibe die Kopfzeile
        csv_writer.writerow(
            ["Airport Code", "Latitude", "Longitude", "City", "Country"]
        )

        # Schreibe die Datenzeilen
        csv_writer.writerows(data)


convert_to_csv(input_file_path, output_file_path)

print(f"Die CSV-Datei wurde erfolgreich unter '{output_file_path}' gespeichert.")
