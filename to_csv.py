from openpyxl import load_workbook
from openpyxl.cell import Cell, MergedCell

book = load_workbook("Revise.xlsx")

sheet = book["Facts"]

rows = []

cs = list(sheet.columns)

columns = []
for column in cs:
    c_list = []
    for cell in column:
        if isinstance(cell, MergedCell):
            c_list.append(c_list[-1])
        elif isinstance(cell, Cell):
            if cell.value and cell.value != "-":
                c_list.append(cell.value)
            else:
                c_list.append("")
        else:
            raise Exception("???")
    columns.append(c_list)

print(len(columns))
print([len(x) for x in columns])

with open("Facts.csv", "w") as output:
    for i in range(1, len(columns[0])):
        output.write(";".join(columns[j][i] for j in range(len(columns))))
        output.write("\n")
