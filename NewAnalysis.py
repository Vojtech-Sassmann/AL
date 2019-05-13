import base64
import binascii
import csv
import operator
import re

import DataPreparation


def decode_base64(data, altchars=b'+/'):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    data = re.sub(rb'[^a-zA-Z0-9%s]+' % altchars, b'', data)  # normalize
    missing_padding = len(data) % 4
    if missing_padding:
        data += b'='* (4 - missing_padding)
    return base64.b64decode(data, altchars)


def process_line(line):



def main():
    with open('./res/originalData/data.csv', newline='', encoding='UTF-8') as f:
        reader = csv.DictReader(f, delimiter=';')

        correct = 0

        incorrect_items = {}
        all_items = {}
        incorrect_ids = []

        for line in reader:
            if int(line['id']) < 6000:
                continue
            if not all_items.keys().__contains__(line['item']):
                all_items[line['item']] = 0
            all_items[line['item']] += 1

            try:
                decoded_solution = DataPreparation.b64decode(line['answer'])
                process_line(line)
                correct += 1
            except Exception as e:
                pass
                #
                #
                # with open('./res/output/err.csv', 'a', encoding='UTF-8') as output:
                #     print(
                #         line['id'],
                #         ";",
                #         line['user'],
                #         ";",
                #         line['item'],
                #         ";",
                #         line['answer'],
                #         ";",
                #         line['correct'],
                #         ";",
                #         line['moves'],
                #         ";",
                #         line['responseTime'],
                #         ";",
                #         line['time'],
                #         file=output,
                #         sep='')
                #
                # print(line['answer'])
                # print(e.args)
                # incorrect_ids.append(line['id'])
                # item_id = line['item']
                # if not incorrect_items.keys().__contains__(item_id):
                #     incorrect_items[item_id] = 1
                # else:
                #     incorrect_items[item_id] += 1
                # continue

        print("Correct: ", correct)
        print("Incorrect: ", len(incorrect_ids))
        print(incorrect_ids)
        print(incorrect_items)
        print(len(incorrect_ids))
        print(incorrect_items.values())

        sorted_items = sorted(incorrect_items.items(), key=lambda x: (-x[1], x[0]))
        print(sorted_items)
        for v, k in sorted_items:
            print("Item:", v, ", Answers:", all_items[v], ", Incorrect:", k, ", Percentage: ", str(k/all_items[v] * 100))

if __name__ == '__main__':
    main()
