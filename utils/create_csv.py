import csv
from utils import common

CSV_FILE = './files/keywords.csv'


def vocab_csv():
    counter = 0
    header = ['elemento', 'tags']
    with open(CSV_FILE, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()

        for _ in common.CLASSES:
            writer.writerow(
                {'elemento': common.BASE_CLASS[counter], 'tags': common.KEYWORDS
                 [counter]})
            counter += 1
