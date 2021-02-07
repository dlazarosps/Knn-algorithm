import argparse
import csv
import math

def read_data(path_file):
    arr_train = []
    for record in csv.DictReader(open(path_file)):
        arr_train.append(record)    
    return arr_train


def distance(p, q, mode = 'euclidean'):
    pd = p.copy()
    qd = q.copy()
    pd.pop('target', None)
    qd.pop('target', None)
    pd.pop('dist', None)
    qd.pop('dist', None)
    
    soma = 0

    for key, value in pd.items():
        diff = float(value) - float(qd[key])
        if mode == 'euclidean':
            soma += pow(diff, 2)
        else:
            soma += abs(diff)

    if mode == 'euclidean':
        ret = math.sqrt(soma)
    else:
        ret = soma

    return ret


def most_frequent(arr): 
    return max(set(arr), key = arr.count)


def knn(kparam, mode, train, test):
    for ts in range(len(test)):
        for tr in range(len(train)):
            dist = distance(train[tr], test[ts], mode)
            train[tr]['dist'] = dist

        train = sorted(train, key=lambda k: k['dist'], reverse=False)
        
        neighbors = []
        for i in range(0, kparam):
            neighbors.append(train[i]['target'])

        test[ts]['predicted'] = most_frequent(neighbors)

def print_results(test, qnt = None):
    if qnt is None:
        qnt = len(test)
        
    for i in range(0, qnt):
        print("{} : target {} / predicted: {}".format(i, test[i]['target'], test[i]['predicted']))


def argparser():
    parser = argparse.ArgumentParser(description = 'K-NN Implementation.')

    parser.add_argument('-k', 
        '--kparam', action = 'store', dest = 'kparam',
        default = 1, required = True, type=int,
        help = 'Number of neighbors')
    
    parser.add_argument('-p', 
        '--print', action = 'store', dest = 'print_qnt',
        default = None, required = False, type=int,
        help = 'Number of neighbors')

    parser.add_argument('-tr', 
        '--train', action = 'store', dest = 'train_path',
        default = './Dataset_CancerClassification/Dados_Normalizados/cancer_train.csv', 
        required = False, help = 'train file path')

    parser.add_argument('-ts', 
        '--test', action = 'store', dest = 'test_path',
        default = './Dataset_CancerClassification/Dados_Normalizados/cancer_test.csv', 
        required = False, help = 'train file path')
    
    parser.add_argument('-m', 
        '--mode', action = 'store', dest = 'mode',
        default = 'euclidean', 
        required = False, help = 'distance method')

    return parser.parse_args()


def main():
    args = argparser()

    # xi = {'x': 0.50, 'y': 0.43}
    # xj = {'x': 1.00, 'y': 0.57}

    # print(distance(xi, xj, args.mode))

    arr_train = read_data(args.train_path)
    arr_test = read_data(args.test_path)

    # print(distance(arr_train[0], arr_train[1], args.mode))

    knn(args.kparam, args.mode, arr_train, arr_test)
            
    print_results(arr_test, args.print_qnt)
    

if __name__ == '__main__':
    main()
