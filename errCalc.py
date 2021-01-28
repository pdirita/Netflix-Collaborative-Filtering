from pyspark import SparkFiles
from pyspark import SparkContext
from pyspark import SparkConf
from operator import add



def errorCalc(predRatings, trueRatings, N):
    predPair = predRatings.map(lambda x: ((x[0],x[1]),x[2]))
    truePair = trueRatings.map(lambda x: ((x[0],x[1]),x[2]))
    joined = predPair.join(truePair)
    MAE = joined.map(lambda (k,v): abs(v[0]-v[1])/N)
    RMS = joined.map(lambda (k,v): ((v[0] - v[1]) ** 2) / N)
    return MAE.reduce(add), (RMS.reduce(add)) ** .5

if __name__ == "__main__":
    conf = SparkConf().setAppName("errtest").setMaster("local")
    sc = SparkContext(conf=conf)
    pred = sc.textFile("ToyPred.txt").map(lambda x: [float(i) for i in x.split(",")])
    act = sc.textFile("ToyActual.txt").map(lambda x: [float(i) for i in x.split(",")])
    N = pred.count()
    MAE, RMS = errorCalc(pred, act, N)
    print(MAE)
    print(RMS)