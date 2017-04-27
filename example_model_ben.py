#!/usr/bin/env python

"""
Example classifier on Numerai data using a logistic regression classifier.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""

import numpy as np
from sklearn.metrics import log_loss as ll
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType


def main():
    # Set seed for reproducibility
    np.random.seed(4)

    # Create SparkContext
    spark = SparkSession.builder.appName('Numerai Model').getOrCreate()

    # Load the data from the CSV files
    print('Loading data...')
    training_data   = spark.read.csv('numerai_training_data.csv',   header=True, inferSchema=True)
    tournament_data = spark.read.csv('numerai_tournament_data.csv', header=True, inferSchema=True)
    validation_data = tournament_data.where('data_type = "validation"')
    prediction_data = tournament_data.where('data_type = "test"')
    print("Finished loading data...")


    # Create a LogisticRegression instance. This instance is an Estimator.
    featureCols = [col for col in training_data.columns if col.startswith("feature")]
    featureAssembler = VectorAssembler(inputCols=featureCols, outputCol="features")
    lr = LogisticRegression(labelCol="target")
    #lr = DecisionTreeClassifier(labelCol="target")
    #lr = MultilayerPerceptronClassifier(labelCol="target")
    #lr = NaiveBayes(labelCol="target")
    pipeline = Pipeline(stages=[featureAssembler, lr])

    # We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
    # This will allow us to jointly choose parameters for all Pipeline stages.
    # A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    # Logistic Regression
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.maxIter, [3, 5, 10]) \
        .addGrid(lr.regParam, [0.2, 0.1, 0.05, 0.01]) \
        .addGrid(lr.elasticNetParam, [0, 0.5, 1]) \
        .addGrid(lr.aggregationDepth, [2, 3]) \
        .build()

    # Decision Tree
    # paramGrid = ParamGridBuilder() \
    #     .addGrid(lr.maxDepth, [3, 5]) \
    #     .addGrid(lr.maxBins, [32, 16, 8]) \
    #     .addGrid(lr.minInfoGain, [0, 0.5, 0.2]) \
    #     .build()
    #     #.addGrid(lr.minInstancesPerNode, [1, 2, 3]) \
    #     #.build()

    # MLP
    # paramGrid = ParamGridBuilder() \
    #     .addGrid(lr.maxIter, [100]) \
    #     .addGrid(lr.blockSize, [4]) \
    #     .addGrid(lr.layers, [[2,2,2]]) \
    #     .addGrid(lr.seed, [123]) \
    #     .build()

    # Naive Bayes
    # paramGrid = ParamGridBuilder() \
    #     .addGrid(lr.modelType, ['multinomial']) \
    #     .addGrid(lr.smoothing, [1.0, 2.0, 0.5]) \
    #     .build()


    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=BinaryClassificationEvaluator(labelCol="target"),
                              numFolds=10)

    # Run cross-validation, and choose the best set of parameters.
    print("Training....")
    cvModel = crossval.fit(training_data)

    # The trained model will now make predictions on the numerai_tournament_data
    # The model returns two columns: [probability of 0, probability of 1]
    # We are just interested in the probability that the target is 1.
    prob1 = udf(lambda probs: float(probs[1]))

    # Print training summary.
    validation_predictions = cvModel.transform(validation_data)
    validation_target_array = validation_predictions.select("target").rdd.map(lambda x: float(x[0])).collect()
    validation_prediction_array = validation_predictions.select(prob1("probability")).rdd.map(lambda x: float(x[0])).collect()
    print "Validation Score (LogLoss):", ll(validation_target_array, validation_prediction_array)

    # Make predictions.
    print("Predicting...")
    prediction = cvModel.transform(tournament_data)

    # Write predictions.
    print("Writing predictions to predictions.csv")
    filteredPredictions = prediction.select("id", prob1("probability").alias("probability")) \
                                    .toPandas()
    filteredPredictions.to_csv("predictions.csv", index=False)


if __name__ == '__main__':
    main()
