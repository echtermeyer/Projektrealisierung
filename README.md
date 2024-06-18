# Project implementation
[Präsentations PowerPoint](https://sap-my.sharepoint.com/:p:/p/lasse_friedrich/EeQcF4qJAe9Ml--uC-vhVyoBIwPtghgj28sviOe2TDfJeg?e=RGU5s3)
### Shared collaboration files:
| Document          | OneDrive Link                                              |
|-------------------|------------------------------------------------------------|
| Projektauftrag    | [View & Edit](https://1drv.ms/w/s!AvZXGwwhrAo8ldw-6gRVStEKGVz88w) |
| Lasten- & Pflichtenheft        | [View & Edit](https://1drv.ms/w/s!AvZXGwwhrAo8ldw_qo5kH2ZkxTQyug) |
| Präsentation 24.05.2024        | [View & Edit](https://1drv.ms/p/s!AvZXGwwhrAo8lf85jEm3VP-JApFspA) |


### Installation

Create your virtual environment using:

```bash
virtualenv -p 3.10 .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the pre-commit hook:

```bash
pre-commit install
```
After that on windows at least you should use git commands only via venv shell, since otherwise pytest can not be executed. (Lasse)

Run python scripts (no `.py` extension):

```bash
python -m src.your_script
```

### Load data

The given CSV files contain errors and cannot be used without cleaning. `TripLoader` cleans and preprocesses the data. A preprocessed version is stored in [src/data/](src/data/) with the suffix `_preprocessed`. When starting `TripLoader` for the first time, all cleaning and preprocessing steps must be completet. This might take up to 10 minutes per dataset. This is only the case for the first time loading the data.

```python
from src.loader import TripLoader

trip_loader = TripLoader()
trips_data = trip_loader.trips_ABCD
```

### Setup

Create a .env file with following content. Note: some variables might not be required for your use case. Add them as you go.

```bash
OPENAI_API_KEY=
```
