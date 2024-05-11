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

### Setup

Create a .env file with following content. Note: some variables might not be required for your use case. Add them as you go.

```bash
OPENAI_API_KEY=
```
