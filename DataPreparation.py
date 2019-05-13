import csv
from base64 import b64decode


def main():
    parse_solutions()


def decode_solution(data):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    missing_padding = len(data) % 4
    if missing_padding != 0:
        data += b'=' * (4 - missing_padding)
    return b64decode(data).decode('utf-8')


def save_solutions(solutions):
    for item_id in solutions.keys():
        with open('./res/parsed/correct_v2/' + item_id + '.txt', mode='w') as output_file:
            for solution in solutions[item_id]:
                print(solution.replace("\n","\\n"), file=output_file)


def parse_solutions():
    solutions = {}
    errors = 0
    errs_info = list()
    with open('./res/originalData/umimeprogramovatcz-ipython_log.csv', newline='', encoding="utf8") as input_file:
        reader = csv.DictReader(input_file, delimiter=';')
        for row in reader:

            # old records contain invalid data
            if int(row['id']) > 6000:
                if row['correct'] != '1':
                    continue
                try:
                    solution = decode_solution(row['answer'])

                    if not solutions.__contains__(row['item']):
                        solutions[row['item']] = list()
                    solutions[row['item']].append(solution)
                    # print(solution)
                except Exception as e:
                    errs_info.append((row['id'], row['user'], row['item'], e, row['answer']))
                    errors += 1
    print('Number of errors: ' + str(errors))
    print('Erred: ')
    for err in errs_info:
        print(err)
    print('------------------')

    print(solutions)

    counts = {}

    for item in solutions.keys():
        counts[item] = solutions[item].__len__()

    for item in sorted(counts):
        print("%s: %s" % (item, counts[item]))
    save_solutions(solutions)


if __name__ == '__main__':
    main()
