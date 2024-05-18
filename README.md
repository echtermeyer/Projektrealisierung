# Project implementation

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
