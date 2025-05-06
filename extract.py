from pprint import pprint
from dataclasses import dataclass


@dataclass
class Table:
    name: str
    columns: list[tuple[int, str]]


@dataclass
class DataTable:
    name: str
    columns: dict[str, list[str]]


def extract_segments(headers: list[str]) -> list[tuple[int, int]]:
    segments = []
    seg_start = 0
    i = 0
    while i < len(headers):
        header = headers[i]
        if not header:
            segments.append((seg_start, i))
            while i < len(headers) and (not headers[i]):
                i += 1
            seg_start = i
        i += 1
    if seg_start < len(headers):
        segments.append((seg_start, len(headers)))
    return segments


def to_table(names: list[str], headers: list[str], segment: tuple[int, int]) -> Table:
    a, b = segment
    name = [names[i] for i in range(a, b) if names[i]][0]
    columns = [(i, headers[i]) for i in range(a, b)]
    return Table(name, columns)


def extract_data(data: list[list[str]], table: Table) -> DataTable:
    columns = [(name, []) for _, name in table.columns]

    for y in range(2, len(data)):
        row = data[y]
        these_values = [row[i] for i, _ in table.columns]
        if all(not a for a in these_values):
            break
        [column[1].append(x) for column, x in zip(columns, these_values)]

    return DataTable(table.name, dict(columns))


def main():
    with open("Facts.csv") as file:
        data = [
            line.strip().split(";")[3:] for line in file.readlines() if line.strip()
        ]

    names = data[0]
    headers = data[1]
    segments = extract_segments(headers)
    tables = [to_table(names, headers, segment) for segment in segments]
    datatables = [extract_data(data, table) for table in tables]

    import os
    from os import path
    import json

    try:
        os.mkdir("out")
    except:
        pass
    for datatable in datatables:
        file_path = path.join("out", f"{datatable.name}.json")
        print(f"Writing to {file_path}")
        with open(file_path, "w") as json_file:
            json.dump(datatable.columns, json_file, indent=4)


if __name__ == "__main__":
    main()
