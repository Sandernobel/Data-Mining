import csv

if __name__ == "__main__":

    with open("ODI-2020.csv") as fq:

        subjects = []
        reader = csv.reader(fq, delimiter=';')
        for row in reader:
            subjects.append(row)

    headers = subjects.pop(0)

    print(f"Rows: {len(subjects)}")
    print(f"Attributes: {len(subjects[0])}")