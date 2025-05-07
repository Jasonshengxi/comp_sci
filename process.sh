set -e
python3 to_csv.py "$1"
python3 extract.py
