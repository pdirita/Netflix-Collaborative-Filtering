import re
import sys
import math
from pyspark import SparkFiles
from operator import add
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating


def least_square(input_file, testing_file, rank, num_iterations):
    # set up spark
    sc = SparkContext()

    # load training data
    training_data = sc.textFile(input_file)
    training_ratings = training_data.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

    # Build the recommendation model using Alternating Least Squares
    # rank - number of hidden features to use
    model = ALS.train(training_ratings, rank, num_iterations)

    # load testing data
    testing_data = sc.textFile(testing_file)
    testing_ratings = testing_data.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

    # Evaluate the model on testing data
    test_input_data = testing_ratings.map(lambda p: (p[0], p[1]))
    predictions = model.predictAll(test_input_data).map(lambda r: ((r[0], r[1]), r[2]))
    ratings_and_predictions = testing_ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)

    # calculate error
    MAE = ratings_and_predictions.map(lambda r: abs(r[1][0] - r[1][1])).mean()
    print("Mean average Error = %s" % MAE)

    RMS = ratings_and_predictions.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    print("Root Mean Squared Error = %s" % RMS)

    # on actual testing data
    # Mean Squared Error = 0.824194342785 for 1 feature and 20 iterations
    # Mean Squared Error = 0.718293348878 for 5 features and 20 iterations
    # Mean Squared Error = 0.705395540942 for 7 features and 20 iterations
    # Mean Squared Error = 0.701640861146 8 features and 20 iterations
    # Mean Squared Error = 0.698334315601 9 features and 20 iterations
    # Mean Squared Error = 0.701380964634 for 10 features and 20 iterations
    # Mean Squared Error = 0.707198188114 for 12 features and 20 iterations
    # Mean Squared Error = 0.717564656906 for 15 features and 20 iterations
    # Mean Squared Error = 0.745623109173 for 20 features and 20 iterations
    # Mean Squared Error = 0.809258285479 for 30 features and 20 iterations
    # Mean Squared Error = 0.94837411753 for 50 features and 20 iterations

    # 20 10
    # Mean average Error = 0.667485663126
    # Root Mean Squared Error = 0.750765069907

    # 9 10
    # Mean average Error = 0.650797025538
    # Root Mean Squared Error = 0.702416923804

    # 9 15
    # Mean average Error = 0.651326045639
    # Root Mean Squared Error = 0.703618478322

    # 9 20
    # Mean average Error = 0.648983400112
    # Root Mean Squared Error = 0.698124265778

    # 9 25
    # Mean average Error = 0.64952279987
    # Root Mean Squared Error = 0.699511058023

    # Save and load model
    # model.save(sc, "target/tmp/myCollaborativeFilter")
    # sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")


# def calcStats(rec):
#     print("stat")
#
#
# def map1(userID,movieID,rating,numRating,sumRating,listA):
#     # if movieID in
#     return (movieID, (userID, rating, numRating, sumRating))


def grouper(l):
    ret = []
    for x in l:
        for y in l:
            if x[0] < y[0]: ret.append(((x[0],y[0]), (x[1] * y[1])))
    return ret


def sq(l):
    ret = [(x[0], x[1]**2) for x in l]
    return ret


def errorCalc(predRatings, trueRatings, N):
    predPair = predRatings.map(lambda x: ((x[0],x[1]),x[2]))
    truePair = trueRatings.map(lambda x: ((x[0],x[1]),x[2]))
    joined = predPair.join(truePair)
    MAE = joined.map(lambda (k,v): abs(v[0]-v[1])/N)
    RMS = joined.map(lambda (k,v): ((v[0] - v[1]) ** 2) / N)
    return MAE.reduce(add), (RMS.reduce(add)) ** .5


def item_item(input_file, testing_file):
    sc = SparkContext()
    training_data = sc.textFile(input_file).map(lambda l: l.split(',')).map(lambda l: (int(l[1]), (int(l[0]), float(l[2]))))
    trainList = training_data.mapValues(lambda x: [x]).reduceByKey(add)
    trainPairs = trainList.flatMap(lambda x: grouper(x[1]))
    trainPairAdd = trainPairs.reduceByKey(add)
    trainListSq = trainList.flatMap(lambda x: sq(x[1]))
    cosDict = trainListSq.reduceByKey(add).map(lambda x: (x[0], x[1]**.5)).collectAsMap()
    simMatrix = trainPairAdd.map(lambda x: (x[0], x[1] / (cosDict[x[0][0]] * cosDict[x[0][1]]))).sortBy(lambda x: x[1], ascending=False).persist()
    k=30
    itemList = training_data.map(lambda x: (x[1])).keys().distinct()
    itemSims = []
    for item in itemList.collect():
        topK = simMatrix.filter(lambda x: item in x[0]).map(lambda x: (x[0][x[0].index(item)-1], x[1])).take(k)
        itemSims.append((item, topK))
    simsRDD = sc.parallelize(itemSims).persist()
    userRatings = training_data.groupByKey().mapValues(lambda x: list(x)).persist()
    testing_data = sc.textFile(testing_file).map(lambda l: l.split(',')).map(lambda l: (int(l[1]), int(l[0])))
    col = testing_data.collect()
    N = len(col)
    preds = []
    trueTest = sc.textFile(testing_file).map(lambda l: l.split(',')).map(lambda l: (int(l[0]), int(l[1]), float(l[2])))
    for movie,user in col:
        numerator = 0
        denom = 0
        try:
            topK = simsRDD.filter(lambda x: x[0] == movie).collect() # [(100, 0.6), ...]
            rated = userRatings.lookup(lambda x: x[0]==user).collect() # [(100, 3.0), (200, 4.0), (300, 3.0)]
        except:
            preds.append([movie,user,0.0])
            continue
        topKMovies = {x[0] : x[1] for x in topK}
        ratedMovies = {x[0] : x[1] for x in rated}
        for i,tMovie in enumerate(topKMovies.keys()):
            if tMovie in ratedMovies.keys():
                numerator += topK[i][1] * ratedMovies[tMovie]
                denom += topK[i][1]
        predicted_rating = 0.0 if denom == 0 else numerator / denom
        preds.append([movie,user,predicted_rating])
        k = len(preds)
        trues = sc.parallelize(trueTest.take(k))
        tempMAE,tempRMS = errorCalc(sc.parallelize(preds),trues,k)
        print("MAE after %s iterations = %s" % (k,tempMAE))
        print("RMS after %s iterations = %s" % (k,tempRMS))
    predsRDD = sc.parallelize(preds)

    MAE,RMS = errorCalc(predsRDD,trueTest,N)
    MAE.saveAsTextFile("s3://netflix-analytics/MAE.txt")
    RMS.saveAsTextFile("s3://netflix-analytics/RMS.txt")
    print(MAE)
    print(RMS)



def main():
    argument_count = len(sys.argv) - 1
    if argument_count == 4:
        input_file = sys.argv[1]
        testing_file = sys.argv[2]
        rank = int(sys.argv[3])
        num_iterations = int(sys.argv[4])
        least_square(input_file, testing_file, rank, num_iterations)
    elif argument_count == 2:
        input_file = sys.argv[1]
        testing_file = sys.argv[2]
        item_item(input_file, testing_file)
    else:
        print("Proper arguments were not passed in.")
        sys.exit(2)


if __name__ == "__main__":
    main()


