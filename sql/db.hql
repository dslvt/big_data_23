DROP DATABASE IF EXISTS projectdb CASCADE;

CREATE DATABASE projectdb; 
USE projectdb;

SET mapreduce.map.output.compress = true; 
SET mapreduce.map.output.compress.codec = org.apache.hadoop.io.compress.SnappyCodec;

CREATE EXTERNAL TABLE accidents STORED AS AVRO LOCATION '/project/accidents' TBLPROPERTIES ('avro.schema.url'='/project/avsc/accidents.avsc');

SELECT * FROM accidents LIMIT 10;


