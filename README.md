# Knn-algorithm

INF01017 - Implementation k-nearest neighbors algorithm  (quantitative)

- Python 3.6.9

## Usage

``` bash
python knn.py --kparam --help [--exec] [--mode] [--normal] [--train] [--test] [--print]
```

### Params

- help
  - list of params
- kparam
  - Integer
    - Number of neighbors (odd)
  - required
  - default = 1
- exec
  - Integer Boolean
    - 1 = repeats the execution decrementing k until it is equal to 1
    - 0 = no repeats
  - default = 0
- mode
  - String
    - mode of distance method
    - 'euclidean' OR 'manhattan'
  - default = 'euclidean'
- normal
  - String
    - shortcut define which dataset path use
    - '' OR 'Nao'
  - default = ''
    - normalized
- train
  - String
    - csv file path with training data
  - default = ''
    - use ./Dataset_CancerClassification/ files
- test
  - String
    - csv file path with testing data
  - default = ''
    - use ./Dataset_CancerClassification/ files
- print
  - Integer
    - debug prints the number of instances prediced
  - default = None
    - all length of test dataset
  - exec parameter has to be off
