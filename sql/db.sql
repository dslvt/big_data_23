\c postgres

DROP DATABASE IF EXISTS project;

CREATE DATABASE project;

\c project;

START TRANSACTION;

CREATE TABLE accidents (
	id TEXT PRIMARY KEY,
	severity smallint NOT NULL,
	start_time timestamp NOT NULL, 
	end_time timestamp NOT NULL, 
	start_lat NUMERIC(9, 6) NOT NULL,
	start_lng NUMERIC(9, 6) NOT NULL,
	end_lat NUMERIC(9, 6),
	end_lng NUMERIC(9, 6),
	distance NUMERIC(7, 3) NOT NULL,
	description TEXT NOT NULL,
	number REAL,
	street TEXT,
	side VARCHAR(1),
	city TEXT,
	county TEXT, 
	state TEXT,
	zipcode TEXT,
	country VARCHAR(2),
	timezone TEXT,
	airport_code VARCHAR(4),
	weather_timestamp timestamp,
	temperature NUMERIC(4, 1),
	wind_chill NUMERIC(4, 1),
	humidity NUMERIC(4, 1),
	pressure NUMERIC(3, 1),
	visibility NUMERIC(4, 1),
	wind_direction TEXT,
	wind_speed REAL, 
	precipitation REAL,
	weather_condition TEXT,
	amenity BOOLEAN NOT NULL,
	bump BOOLEAN NOT NULL,
	crossing BOOLEAN NOT NULL,
	give_away BOOLEAN NOT NULL,
	junction BOOLEAN NOT NULL,
	no_exit BOOLEAN NOT NULL,
	railway BOOLEAN NOT NULL,
	roundabout BOOLEAN NOT NULL,
	station BOOLEAN NOT NULL,
	stop BOOLEAN NOT NULL,
	traffic_calming BOOLEAN NOT NULL,
	traffic_signal BOOLEAN NOT NULL,
	turning_loop BOOLEAN NOT NULL,
	sunrise_sunset TEXT,
	civil_twilight TEXT,
	nautical_twilight TEXT,
	astronomical_twilight TEXT
);

SET datestyle TO iso, ymd;

\COPY accidents FROM '/data/US_Accidents_Dec21_updated.csv' DELIMITER ',' CSV HEADER NULL AS '';

COMMIT;

SELECT * FROM accidents;
