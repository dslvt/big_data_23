import pyspark.sql.functions as F  
from pyspark.sql import SparkSession, Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, VectorIndexer
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier

import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from math import atan, tan, atan2, cos, radians, sin, sqrt


def create_session():
	print('Creating Spark Session')
	spark = SparkSession.builder.appName("BDT Project").master("local[*]").config("hive.metastore.uris", "thrift://sandbox-hdp.hortonworks.com:9083").config("spark.sql.catalogImplementation","hive").config("spark.sql.avro.compression.codec", "snappy").config("spark.jars", "file:///usr/hdp/current/hive-client/lib/hive-metastore-1.2.1000.2.6.5.0-292.jar,file:///usr/hdp/current/hive-client/lib/hive-exec-1.2.1000.2.6.5.0-292.jar").config("spark.jars.packages","org.apache.spark:spark-avro_2.12:3.0.3").enableHiveSupport().getOrCreate()

	sc = spark.sparkContext
	sc.setLogLevel("ERROR")
	print(sc)
	return spark, sc


def read_data():
	accidents = spark.read.format("avro").table('projectdb.accidents')
	accidents.createOrReplaceTempView('accidents')
	accidents = accidents.withColumn("distance", accidents["distance"].cast("float"))
	accidents = accidents.withColumn("humidity", accidents["humidity"].cast("float"))
	accidents = accidents.withColumn("pressure", accidents["pressure"].cast("float"))
	accidents = accidents.withColumn("visibility", accidents["visibility"].cast("float"))
	accidents = accidents.withColumn("wind_chill", accidents["wind_chill"].cast("float"))
	accidents = accidents.withColumn("temperature", accidents["temperature"].cast("float"))
	accidents = accidents.withColumn("start_lat", accidents["start_lat"].cast("float"))
	accidents = accidents.withColumn("start_lng", accidents["start_lng"].cast("float"))
	accidents = accidents.withColumn("end_lat", accidents["end_lat"].cast("float"))
	accidents = accidents.withColumn("end_lng", accidents["end_lng"].cast("float"))

	df_cleared = accidents.na.drop()

	frac = float(50000)/889904
	sample = df_cleared.where(df_cleared.severity == 2).sample(fraction=frac, withReplacement=False, seed=42)
	df_balanced = df_cleared.where(df_cleared.severity != 2).union(sample)

	df_balanced = df_balanced.withColumn("start_time", (F.col("start_time")/1000).cast("timestamp"))
	df_balanced = df_balanced.withColumn("end_time", (F.col("end_time")/1000).cast("timestamp"))
	return df_balanced


def datetime_features_extraction(df_balanced):
	df_balanced = df_balanced.withColumn("DayOfWeek", F.dayofweek("start_time"))
	df_balanced = df_balanced.withColumn("Hour", F.hour("start_time"))
	df_balanced = df_balanced.withColumn('accident_duration', (F.unix_timestamp('End_Time') - F.unix_timestamp('Start_Time')))


def read_ucities():
	cities = pd.read_csv('uscities.csv')
	# Sampling large cities
	cities = cities[cities['population'] >= 5000]
	return cities


def vincenty_inverse(coord1, coord2, maxIter=10, tol=10**-12):
    a = 6378137.0  # radius at equator in meters (WGS-84)
    f = 1 / 298.257223563  # flattening of the ellipsoid (WGS-84)
    b = (1 - f) * a
    
    phi_1, L_1 = coord1  # (lat=L_?,lon=phi_?)
    phi_2, L_2 = coord2
    
    u_1 = atan((1 - f) * tan(radians(phi_1)))
    u_2 = atan((1 - f) * tan(radians(phi_2)))
    
    L = radians(L_2 - L_1)
    
    Lambda = L  # set initial value of lambda to L
    
    sin_u1 = sin(u_1)
    cos_u1 = cos(u_1)
    sin_u2 = sin(u_2)
    cos_u2 = cos(u_2)
    
    # --- BEGIN ITERATIONS -----------------------------+
    iters = 0
    for i in range(0, maxIter):
        iters += 1
        
        cos_lambda = cos(Lambda)
        sin_lambda = sin(Lambda)
        sin_sigma = sqrt(
            (cos_u2 * sin(Lambda)) ** 2
            + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda) ** 2
        ) + 1e-8
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda
        sigma = atan2(sin_sigma, cos_sigma)
        sin_alpha = (cos_u1 * cos_u2 * sin_lambda) / sin_sigma
        cos_sq_alpha = 1 - sin_alpha**2
        cos2_sigma_m = cos_sigma - ((2 * sin_u1 * sin_u2) / cos_sq_alpha)
        C = (f / 16) * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))
        Lambda_prev = Lambda
        Lambda = L + (1 - C) * f * sin_alpha * (
            sigma
            + C
            * sin_sigma
            * (cos2_sigma_m + C * cos_sigma * (-1 + 2 * cos2_sigma_m**2))
        )
        
        # successful convergence
        diff = abs(Lambda_prev - Lambda)
        if diff <= tol:
            break
            
    u_sq = cos_sq_alpha * ((a**2 - b**2) / b**2)
    A = 1 + (u_sq / 16384) * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    B = (u_sq / 1024) * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
    delta_sig = (
        B
        * sin_sigma
        * (
            cos2_sigma_m
            + 0.25
            * B
            * (
                cos_sigma * (-1 + 2 * cos2_sigma_m**2)
                - (1 / 6)
                * B
                * cos2_sigma_m
                * (-3 + 4 * sin_sigma**2)
                * (-3 + 4 * cos2_sigma_m**2)
            )
        )
    )
    
    return b * A * (sigma - delta_sig)


def retrieve_coordinates(df_balanced):
	cities_lat = cities['lat'].values
	cities_lng = cities['lng'].values
	cities_cor = list(zip(cities_lat, cities_lng))
	
	start_lat = df_balanced.select('start_lat').toPandas()['start_lat'].tolist()
	start_lng = df_balanced.select('start_lng').toPandas()['start_lng'].tolist()
	
	end_lat = df_balanced.select('end_lat').toPandas()['end_lat'].tolist()
	end_lng = df_balanced.select('end_lng').toPandas()['end_lng'].tolist()
	
	distances = []
	for i in range(len(start_lng)):
		distances.append(vincenty_inverse((start_lng[i], start_lat[i]), (end_lng[i], end_lat[i])))
	
	distances_df = spark.createDataFrame([(l,) for l in distances], ['accident_distance'])

	df_balanced = df_balanced.withColumn("row_idx", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
	distances_df = distances_df.withColumn("row_idx", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
	df_balanced = df_balanced.join(distances_df, df_balanced.row_idx == distances_df.row_idx).\
             drop("row_idx")
	
	return cities_cor, start_lat, start_lng, end_lat, end_lng
	

def calc_distances_to_cities(df_balanced, cities_cor, start_lat, start_lng, end_lat, end_lng):
	cities_cor_rad = np.array([[radians(x[0]), radians(x[1])] for x in cities_cor ])
	tree = BallTree(cities_cor_rad, metric = 'haversine')
	min_distance_start = []
	min_distance_end = []
	earth_radius = 6371

	for i in range(len(start_lat)):
		dist_start = 1e9
		dist_end = 1e9
		
		start_result = tree.query([(radians(start_lat[i]),radians(start_lng[i]))])
		end_result = tree.query([(radians(end_lat[i]),radians(end_lng[i]))])
				
		min_distance_start.append((start_result[0][0] * earth_radius)[0])
		min_distance_end.append((end_result[0][0] * earth_radius)[0])

	distances_df = spark.createDataFrame([(float(s),float(e),) for s, e in zip(min_distance_start, min_distance_end)], ['start_dist_city', 'end_dist_city'])
	df_balanced = df_balanced.withColumn("row_idx", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
	distances_df = distances_df.withColumn("row_idx", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
	df_balanced = df_balanced.join(distances_df, df_balanced.row_idx == distances_df.row_idx).\
				drop("row_idx")


def get_columns_for_model(df_balanced):
	to_drop = ['number', 'street', 'side', 'city', 'county', 'zipcode', 'country', 'timezone', 'airport_code', 'weather_timestamp', 'weather_condition', 'start_time', 'end_time', 'start_lat', 'start_lng', 'end_lat', 'end_lng', 'description', 'id', 'wind_direction', 'severity']

	to_encode = ['state', 'sunrise_sunset', 'civil_twilight', 'nautical_twilight', 'astronomical_twilight']

	numerical = [c for c in df_balanced.columns if c not in to_drop + to_encode]
	
	return to_encode, numerical


def encode_and_split(df_balanced, to_drop, to_encode, numerical):
	indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c)).setHandleInvalid("skip") for c in to_encode]

	encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol="{0}_encoded".format(indexer.getOutputCol())) for indexer in indexers]


	assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders] + numerical, outputCol= "features")

	encoding_pipeline = Pipeline(stages=indexers + encoders + [assembler])

	encoding_model=encoding_pipeline.fit(df_balanced)
	
	data = encoding_model.transform(df_balanced)
	data = data.select(["features", "severity"])

	featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
	transformed = featureIndexer.transform(data)

	(trainingData, testData) = transformed.randomSplit([0.6, 0.4])
	return trainingData, testData


if __name__ == 'main':
	spark, sc = create_session()
	df_balanced = read_data()
	datetime_features_extraction(df_balanced)
	cities = read_ucities()
	cities_cor, start_lat, start_lng, end_lat, end_lng = retrieve_coordinates(df_balanced)
	calc_distances_to_cities(df_balanced, cities_cor, start_lat, start_lng, end_lat, end_lng)
	to_encode, numerical = get_columns_for_model(df_balanced)
	trainingData, testData = encode_and_split(df_balanced, to_encode, numerical)
	DT_model = DecisionTreeClassifier.load('./models/DecisionTreeClassifier_model')
	RF_model = RandomForestClassifier.load('./models/GBTClassifier_model')
	DT_predictions = DT_model.transform(testData)
	RF_predictions = RF_model.transform(testData)
	RF_predictions.coalesce(1).select("prediction",'severity').write.mode("overwrite").format("csv").option("sep", ",").option("header","true").csv("/RF_predictions.csv")
	DT_predictions.coalesce(1).select("prediction",'severity').write.mode("overwrite").format("csv").option("sep", ",").option("header","true").csv("/DT_predictions.csv")