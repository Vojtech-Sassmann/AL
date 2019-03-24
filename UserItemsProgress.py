import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
from base64 import b64decode


def decode_solution(data):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    missing_padding = len(data) % 4
    if missing_padding != 0:
        data += b'=' * (4 - missing_padding)
    return b64decode(data).decode('utf-8')


def save_data(data):
    for user_id in data.keys():
        for item_id in data[user_id].keys():
            with open('./res/progress/raw/all.txt', mode='a') as output_file:
                print(user_id, file=output_file)
                for solution in data[user_id][item_id]:
                    print(solution.replace("\n", "\\n"), file=output_file)
                print("###", file=output_file)


def save_only_correct(correct_data):
    for item_id in correct_data.keys():
        with open('./res/progress/raw/correct/all.txt', mode='a') as output_file:
        # with open('./res/progress/raw/correct/' + str(item_id) + '.txt', mode='w') as output_file:
            for solution_set in correct_data[item_id]:
                print("1111", file=output_file)
                for solution in solution_set:
                    print(solution.replace("\n", "\\n"), file=output_file)
                print("###", file=output_file)


def parse_solutions():
    solutions = {}
    items_data = {}
    correct_data = {}
    errors = 0
    errs_info = list()
    with open('./res/originalData/umimeprogramovatcz-ipython_log.csv', newline='', encoding="utf8") as input_file:
        reader = csv.DictReader(input_file, delimiter=';')
        for row in reader:

            # old records contain invalid data
            if int(row['id']) > 6000:
                # if row['correct'] != '1':
                #     continue
                try:
                    solution = decode_solution(row['answer'])
                    user_id = row['user']
                    item_id = row['item']

                    if not items_data.__contains__(user_id):
                        items_data[user_id] = {}
                    if not items_data[user_id].__contains__(item_id):
                        items_data[user_id][item_id] = list()

                    items_data[user_id][item_id].append(solution)

                    if row['correct'] == '1':
                        if not correct_data.keys().__contains__(item_id):
                            correct_data[item_id] = list()
                        correct_data[item_id].append(items_data[user_id][item_id])
                        items_data[user_id][item_id] = list()

                except Exception as e:
                    errs_info.append((row['id'], row['user'], row['item'], e, row['answer']))
                    errors += 1
    print('Number of errors: ' + str(errors))
    print('Erred: ')
    # for err in errs_info:
    #     print(err)
    print('------------------')

    print(solutions)

    counts = {}

    for item in solutions.keys():
        counts[item] = solutions[item].__len__()

    for item in sorted(counts):
        print("%s: %s" % (item, counts[item]))

    # save_data(items_data)
    save_only_correct(correct_data)


def show_chart(data, x_title, y_title):
    objects = sorted(data.keys())
    y_pos = np.arange(len(objects))

    counts = []

    for k in sorted(data.keys()):
        counts.append(data[k])

    plt.bar(y_pos, counts, align='center', alpha=0.5)
    plt.xticks(y_pos, objects, rotation='vertical')
    plt.ylabel(y_title)
    plt.title(x_title)

    # plt.show()


def analyze_count_of_submissions(file):
    data = {}
    with open(file, newline='', encoding="utf8") as input_file:
        line = True  # not a very nice solution

        while line:
            line = input_file.readline()
            counter = 0
            try:
                user_id = int(line)
            except:
                break
            line = input_file.readline()
            while not line.startswith("###") and line:
                counter += 1
                line = input_file.readline()

            if not data.__contains__(counter):
                data[counter] = 0
            data[counter] += 1
    for k in sorted(data.keys()):
        print(k, ":", data[k])

    return data


def main():
    # parse_solutions()
    data_all_all = analyze_count_of_submissions('./res/progress/raw/all.txt')
    data_all_correct = analyze_count_of_submissions('./res/progress/raw/correct/all.txt')

    show_chart(data_all_all, "All items / All submissions", "Attempts")
    plt.savefig('./res/figs/all_all.svg')
    plt.show()
    show_chart(data_all_correct, "All items / Only correct", "Attempts")
    plt.savefig('./res/figs/all_correct.svg')
    plt.show()

    for i in range(1, 74):
        try:
            data = analyze_count_of_submissions('./res/progress/raw/' + str(i) + '.txt')
            show_chart(data, str(i), "Attempts")
            plt.savefig('./res/figs/' + str(i) + '.svg')
            plt.show()
        except:
            pass


if __name__ == '__main__':
    main()
