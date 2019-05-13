import pandas as pd


def analyze_solutions(df, id):
    results = []
    distances_total = 0
    for solution, distances in df.iterrows():
        distances_sum = 0
        for distance in distances:
            distances_sum += distance

        distances_total += distances_sum
        results.append((solution, distances_sum))

    # sorted_items = sorted(result.items(), key=lambda x: (-x[1], x[0]))

    solution_rate = []

    average = distances_total / float(len(results))

    for solution, distance in results:
        solution_rate.append((solution, distance / average))

    has = False
    with open('./res/output/rare_v2/' + id, 'w', encoding='UTF-8') as f:
        for solution, rate in sorted(solution_rate, key=lambda x: x[1], reverse=True):
            if not has:
                has = True
                print("Id: ", id, " rate: ", rate)
            print(rate, file=f)
            print(solution, file=f)



def analyze_file(file_name, id):
    df = pd.read_pickle(path + "distances/" + file_name)
    analyze_solutions(df, id)



path = "./res/solutiongroups/dbscan/correct_v2/"
DBSCAN_MIN_SIZE = 5
DBSCAN_EPSILON = 1


if __name__ == '__main__':
    for i in range(1, 34):
        try:
            analyze_file(str(i) + '.pkl', str(i))
        except Exception as e:
            print(e)

