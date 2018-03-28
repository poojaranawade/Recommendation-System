import pandas as pd
import numpy
import csv
from recommendation import cosine_sim, pearson_corelation,get_similar_movies,item_based

if __name__ == '__main__':
    # =============================================================================
    # reading data from files
    # =============================================================================
    with open('train.txt', 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split("\t") for line in stripped if line)
        with open('train.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(lines)

    train_data = pd.read_csv('train.csv', header=None)
    train_mat = train_data.as_matrix()
    similar_movies,cosine_movies=get_similar_movies(train_mat)
    
    # =============================================================================
    # test5.text
    # =============================================================================
    with open('test5.txt', 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(" ") for line in stripped if line)
        with open('test5.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(lines)

    test_data = pd.read_csv('test5.csv', header=None)
    test_mat = numpy.zeros(shape=(100, 1000))
    predict = {k: [] for k in range(100)}
    for row in test_data.iterrows():
        test_mat[row[1][0] - 201][row[1][1] - 1] = row[1][2]
        if row[1][2] == 0:
            predict[row[1][0] - 201].append(row[1][1] - 1)

    with open('testResC5.txt', 'w') as res_file:
        for i, row in enumerate(test_mat):
            print("\ni", i+201, end=" ")
            for index, col in enumerate(row):
                cosine_sim_rating,pearson_rating=0,0
                if col == 0 and index in predict[i]:
                    cosine_sim_rating = cosine_sim(i, index, row, train_mat)
                    pearson_rating = pearson_corelation(i, index, row, train_mat)
                    row[index]=(cosine_sim_rating+pearson_rating)/2
#                    rating=item_based(index,row,similar_movies,cosine_movies,train_mat)
                    res_file.write("%d %d %d\n" % ((i + 201), index + 1, row[index]))
    # =============================================================================
    # test10.text
    # =============================================================================
    with open('test10.txt', 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(" ") for line in stripped if line)
        with open('test10.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(lines)

    test_data = pd.read_csv('test10.csv', header=None)
    test_mat = numpy.zeros(shape=(100, 1000))
    predict = {k: [] for k in range(100)}
    for row in test_data.iterrows():
        test_mat[row[1][0] - 301][row[1][1] - 1] = row[1][2]
        if row[1][2] == 0:
            predict[row[1][0] - 301].append(row[1][1] - 1)

    with open('testResC10.txt', 'w') as res_file:
        for i, row in enumerate(test_mat):
            print("\ni", i+301, end=" ")
            for index, col in enumerate(row):
                cosine_sim_rating,pearson_rating=0,0
                if col == 0 and index in predict[i]:
                    cosine_sim_rating = cosine_sim(i, index, row, train_mat)
                    pearson_rating = pearson_corelation(i, index, row, train_mat)
                    row[index]=(cosine_sim_rating+pearson_rating)/2
#                    row[index]=item_based(index,row,similar_movies,cosine_movies,train_mat)
                    res_file.write("%d %d %d\n" % ((i + 301), index + 1, row[index]))
    # =============================================================================
    # test20.text
    # =============================================================================
    with open('test20.txt', 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(" ") for line in stripped if line)
        with open('test20.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(lines)

    test_data = pd.read_csv('test20.csv', header=None)
    test_mat = numpy.zeros(shape=(100, 1000))
    predict = {k: [] for k in range(100)}
    for row in test_data.iterrows():
        test_mat[row[1][0] - 401][row[1][1] - 1] = row[1][2]
        if row[1][2] == 0:
            predict[row[1][0] - 401].append(row[1][1] - 1)

    with open('testResC20.txt', 'w') as res_file:
        for i, row in enumerate(test_mat):
            print("\ni", i+401, end=" ")
            for index, col in enumerate(row):
                cosine_sim_rating,pearson_rating=0,0
                if col == 0 and index in predict[i]:
                    cosine_sim_rating = cosine_sim(i, index, row, train_mat)
                    pearson_rating = pearson_corelation(i, index, row, train_mat)
                    row[index]=(cosine_sim_rating+pearson_rating)/2
#                    row[index]=item_based(index,row,similar_movies,cosine_movies,train_mat)
                    res_file.write("%d %d %d\n" % ((i + 401), index + 1, row[index]))