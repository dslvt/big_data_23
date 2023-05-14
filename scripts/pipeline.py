"""
Pipeline for data preprocessing anmd modeling
"""
from math import atan, tan, atan2, cos, radians, sin, sqrt

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    VectorIndexer,
)
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier

import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

RADIUS = 6378137.0
FLATTENING = 1 / 298.257223563
B_VAL = (1 - FLATTENING) * RADIUS

def create_session():
    """
    Creating sesstion.
    """

    print("Creating Spark Session")
    line_url1 = "file:///usr/hdp/current/hive-client/lib/hive-metastore-1.2.1000.2.6.5.0-292.jar"
    line_url2 = (
        "file:///usr/hdp/current/hive-client/lib/hive-exec-1.2.1000.2.6.5.0-292.jar"
    )
    spark_v = (
        SparkSession.builder.appName("BDT Project")
        .master("local[*]")
        .config("hive.metastore.uris", "thrift://sandbox-hdp.hortonworks.com:9083")
        .config("spark.sql.catalogImplementation", "hive")
        .config("spark.sql.avro.compression.codec", "snappy")
        .config(
            "spark.jars",
            f"{line_url1},{line_url2}",
        )
        .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.0.3")
        .enableHiveSupport()
        .getOrCreate()
    )

    spack_context_s = spark_v.sparkContext
    spack_context_s.setLogLevel("ERROR")
    print(spack_context_s)
    return spark_v, spack_context_s


def read_data():
    """
    Reading full data
    """
    accidents = spark.read.format("avro").table("projectdb.accidents")
    accidents.createOrReplaceTempView("accidents")
    accidents = accidents.withColumn("distance", accidents["distance"].cast("float"))
    accidents = accidents.withColumn("humidity", accidents["humidity"].cast("float"))
    accidents = accidents.withColumn("pressure", accidents["pressure"].cast("float"))
    accidents = accidents.withColumn(
        "visibility", accidents["visibility"].cast("float")
    )
    accidents = accidents.withColumn(
        "wind_chill", accidents["wind_chill"].cast("float")
    )
    accidents = accidents.withColumn(
        "temperature", accidents["temperature"].cast("float")
    )
    accidents = accidents.withColumn("start_lat", accidents["start_lat"].cast("float"))
    accidents = accidents.withColumn("start_lng", accidents["start_lng"].cast("float"))
    accidents = accidents.withColumn("end_lat", accidents["end_lat"].cast("float"))
    accidents = accidents.withColumn("end_lng", accidents["end_lng"].cast("float"))

    df_cleared = accidents.na.drop()

    frac = float(50000) / 889904
    sample = df_cleared.where(df_cleared.severity == 2).sample(
        fraction=frac, withReplacement=False, seed=42
    )
    data = df_cleared.where(df_cleared.severity != 2).union(sample)

    data = data.withColumn(
        "start_time", (F.col("start_time") / 1000).cast("timestamp")
    )
    data = data.withColumn(
        "end_time", (F.col("end_time") / 1000).cast("timestamp")
    )
    return data


def datetime_features_extraction(data):
    """
    Extracting time based features
    """
    data = data.withColumn("DayOfWeek", F.dayofweek("start_time"))
    data = data.withColumn("Hour", F.hour("start_time"))
    data = data.withColumn(
        "accident_duration",
        (F.unix_timestamp("End_Time") - F.unix_timestamp("Start_Time")),
    )


def read_ucities():
    """
    Reading additional data
    """

    cities_df = pd.read_csv("uscities.csv")
    # Sampling large cities
    cities_df = cities_df[cities_df["population"] >= 5000]
    return cities_df


def vincenty_inverse(coord1, coord2, max_iter=10, tol=10**-12):
    """
    Calculating distance between two points
    """
    u_1 = atan((1 - FLATTENING) * tan(radians(coord1[0])))
    u_2 = atan((1 - FLATTENING) * tan(radians(coord2[0])))

    l_val = radians(coord2[1] - coord1[1])

    lambda_v = l_val  # set initial value of lambda to L

    # --- BEGIN ITERATIONS -----------------------------+
    iters = 0
    for _ in range(0, max_iter):
        iters += 1

        cos_lambda = cos(lambda_v)
        sin_lambda = sin(lambda_v)
        sin_sigma = (
            sqrt(
                (cos(u_2) * sin(lambda_v)) ** 2
                + (cos(u_1) * sin(u_2) - sin(u_1) * cos(u_2) * cos_lambda) ** 2
            )
            + 1e-8
        )
        cos_sigma = sin(u_1) * sin(u_2) + cos(u_1) * cos(u_2) * cos_lambda
        sigma = atan2(sin_sigma, cos_sigma)
        sin_alpha = (cos(u_1) * cos(u_2) * sin_lambda) / sin_sigma
        cos_sq_alpha = 1 - sin_alpha**2
        cos2_sigma_m = cos_sigma - ((2 * sin(u_1) * sin(u_2)) / cos_sq_alpha)
        c_val = (FLATTENING / 16) * cos_sq_alpha * (4 + FLATTENING * (4 - 3 * cos_sq_alpha))
        lambda_prev = lambda_v
        lambda_v = l_val + (1 - c_val) * FLATTENING * sin_alpha * (
            sigma
            + c_val
            * sin_sigma
            * (cos2_sigma_m + c_val * cos_sigma * (-1 + 2 * cos2_sigma_m**2))
        )

        # successful convergence
        diff = abs(lambda_prev - lambda_v)
        if diff <= tol:
            break

    u_sq = cos_sq_alpha * ((RADIUS**2 - B_VAL**2) / B_VAL**2)
    a_pos = 1 + (u_sq / 16384) * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    b_pos = (u_sq / 1024) * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
    delta_sig = (
        b_pos
        * sin_sigma
        * (
            cos2_sigma_m
            + 0.25
            * b_pos
            * (
                cos_sigma * (-1 + 2 * cos2_sigma_m**2)
                - (1 / 6)
                * b_pos
                * cos2_sigma_m
                * (-3 + 4 * sin_sigma**2)
                * (-3 + 4 * cos2_sigma_m**2)
            )
        )
    )

    return B_VAL * a_pos * (sigma - delta_sig)


def retrieve_coordinates(data):
    """
    Getting nearest city
    """

    cities_lat = cities["lat"].values
    cities_lng = cities["lng"].values
    city_cords = list(zip(cities_lat, cities_lng))

    lt_start = data.select("start_lat").toPandas()["start_lat"].tolist()
    lg_start = data.select("start_lng").toPandas()["start_lng"].tolist()

    lt_end = data.select("end_lat").toPandas()["end_lat"].tolist()
    lg_end = data.select("end_lng").toPandas()["end_lng"].tolist()

    distances = []
    for i, ln_start in enumerate(lg_start):
        distances.append(
            vincenty_inverse((ln_start, lt_start[i]), (lg_end[i], lt_end[i]))
        )

    distances_df = spark.createDataFrame(
        [(l,) for l in distances], ["accident_distance"]
    )

    data = data.withColumn(
        "row_idx", F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))
    )
    distances_df = distances_df.withColumn(
        "row_idx", F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))
    )
    data = data.join(
        distances_df, data.row_idx == distances_df.row_idx
    ).drop("row_idx")

    return city_cords, lt_start, lg_start, lt_end, lg_end


def calc_distances_to_cities(data, city_cors, cords):
    """
    Geting nearest city
    """
    lt_start, lg_start, lt_end, lg_end = cords
    cities_cor_rad = np.array([[radians(x[0]), radians(x[1])] for x in city_cors])
    tree = BallTree(cities_cor_rad, metric="haversine")
    min_distance_start = []
    min_distance_end = []

    for i, start_lt in enumerate(lt_start):
        min_distance_start.append(
            (tree.query([(radians(start_lt), radians(lg_start[i]))])[0][0] * 6371)[0]
        )
        min_distance_end.append(
            (tree.query([(radians(lt_end[i]), radians(lg_end[i]))])[0][0] * 6371)[0]
        )

    distances_df = spark.createDataFrame(
        [
            (
                float(s),
                float(e),
            )
            for s, e in zip(min_distance_start, min_distance_end)
        ],
        ["start_dist_city", "end_dist_city"],
    )
    data = data.withColumn(
        "row_idx", F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))
    )
    distances_df = distances_df.withColumn(
        "row_idx", F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))
    )
    data = data.join(distances_df, data.row_idx == distances_df.row_idx).drop("row_idx")


def get_columns_for_model(data):
    """
    Defining columns to preprocessing
    """
    to_drop = [
        "number",
        "street",
        "side",
        "city",
        "county",
        "zipcode",
        "country",
        "timezone",
        "airport_code",
        "weather_timestamp",
        "weather_condition",
        "start_time",
        "end_time",
        "start_lat",
        "start_lng",
        "end_lat",
        "end_lng",
        "description",
        "id",
        "wind_direction",
        "severity",
    ]

    to_encode = [
        "state",
        "sunrise_sunset",
        "civil_twilight",
        "nautical_twilight",
        "astronomical_twilight",
    ]

    numerical = [c for c in data.columns if c not in to_drop + to_encode]

    return to_encode, numerical


def encode_and_split(data, columns_to_encode, numerical):
    """
    Part for encoding infomation, and splitting it into parts
    """
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_indexed").setHandleInvalid("skip")
        for c in columns_to_encode
    ]

    encoders = [
        OneHotEncoder(
            inputCol=indexer.getOutputCol(),
            outputCol=f"{indexer.getOutputCol()}_encoded",
        )
        for indexer in indexers
    ]

    assembler = VectorAssembler(
        inputCols=[encoder.getOutputCol() for encoder in encoders] + numerical,
        outputCol="features",
    )

    encoding_pipeline = Pipeline(stages=indexers + encoders + [assembler])

    encoding_model = encoding_pipeline.fit(data)

    data = encoding_model.transform(data)
    data = data.select(["features", "severity"])

    feature_indexer = VectorIndexer(
        inputCol="features", outputCol="indexedFeatures", maxCategories=4
    ).fit(data)
    transformed = feature_indexer.transform(data)

    (training_dt, test_dt) = transformed.randomSplit([0.6, 0.4])
    return training_dt, test_dt


if __name__ == "main":
    spark, spack_context = create_session()
    df_balanced = read_data()
    datetime_features_extraction(df_balanced)
    cities = read_ucities()
    cities_cor, start_lat, start_lng, end_lat, end_lng = retrieve_coordinates(
        df_balanced
    )
    calc_distances_to_cities(
        df_balanced, cities_cor, (start_lat, start_lng, end_lat, end_lng)
    )
    col_to_encode, numerical_data = get_columns_for_model(df_balanced)
    training_data, test_data = encode_and_split(
        df_balanced, col_to_encode, numerical_data
    )
    DT_model = DecisionTreeClassifier.load("./models/DecisionTreeClassifier_model")
    RF_model = RandomForestClassifier.load("./models/GBTClassifier_model")
    DT_predictions = DT_model.transform(test_data)
    RF_predictions = RF_model.transform(test_data)
    RF_predictions.coalesce(1).select("prediction", "severity").write.mode(
        "overwrite"
    ).format("csv").option("sep", ",").option("header", "true").csv(
        "/RF_predictions.csv"
    )
    DT_predictions.coalesce(1).select("prediction", "severity").write.mode(
        "overwrite"
    ).format("csv").option("sep", ",").option("header", "true").csv(
        "/DT_predictions.csv"
    )
