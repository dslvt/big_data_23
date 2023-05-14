#!/bin/bash
psql -U postgres -d project -f sql/db.sql
sqoop import-all-tables  -Dmapreduce.job.user.classpath.first=true  --connect jdbc:postgresql://localhost/project   --username postgres  --warehouse-dir /project    --as-avrodatafile   --compression-codec=snappy --m 1
hdfs dfs -put accidents.avsc /project/accidents.avsc
