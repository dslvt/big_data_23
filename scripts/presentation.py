"""
Code for displaying EDA and PDA.
"""

import streamlit as st
import pandas as pd
import altair as alt


def main():
    """
    Streamlit EDA and PDA
    """
    display_main()
    display_twilight()
    display_weather()
    display_poi()
    display_traffic_info()
    display_pda()


def display_main():
    """
    Displaying project info and overall information
    """

    data = pd.read_csv("data/data_prepared.csv")
    target = pd.read_csv("output/EDA_outputs/targer_dist.csv")
    coordinates = pd.read_csv("output/EDA_outputs/coordinates.csv")
    time = pd.read_csv("output/EDA_outputs/times.csv", parse_dates=["start_time"])
    distance = pd.read_csv("output/EDA_outputs/severity_distance.csv")

    st.header("Big Data Project")
    st.write("Authors: Timur Sergeev, Salavat Dinmukhametov")

    st.markdown("**Project description**")
    st.markdown(
        """This is a countrywide traffic accident dataset, which covers 49 states of the 
        United States. The data is continuously being collected from February 2016, using 
        several data providers, including multiple APIs that provide streaming traffic event data. 
        These APIs broadcast traffic events captured by a variety of entities, such 
        as the US and state departments of transportation, law enforcement agencies, 
        traffic cameras, and traffic sensors within the road-networks. Currently, 
        there are about 1.5 million accident records in this dataset. After 
        removing nans and normalizing targets we got 103k samples."""
    )
    st.divider()

    st.header("EDA")

    st.markdown(f"Dataset shape: {data.shape}")

    st.table(data.head(10))
    columns = list(data.columns)
    columns_str = "**Columns:**\n"

    for i in columns:
        columns_str += "- " + i + "\n"

    st.markdown(columns_str)

    st.subheader("Target")
    target = target.set_index("severity")
    st.bar_chart(target)
    st.markdown("Majority of the accident have 2 severity.")
    st.divider()

    st.subheader("Accident position")
    coordinates = coordinates.rename(columns={"start_lat": "lat", "start_lng": "lon"})
    st.markdown(
        "Majority of the accident are near cities, so we added additianl dataset with cities cords."
    )
    st.map(coordinates)

    st.subheader("Distance")
    distance = distance.set_index("severity")
    st.markdown("There are linear correlation with target.")
    st.bar_chart(distance)

    st.subheader("Accident start time")
    time["start_time"] = pd.to_datetime(time["start_time"])
    time = time.set_index("start_time")
    time["count"] = 1
    summary = time.resample("M").sum()
    st.line_chart(summary)
    st.markdown(
        "There are peak in april 2020. Also number of accidents are linearly growing."
    )
    st.divider()


def display_twilight():
    """
    Displaying twilight
    """

    astronomical_twilight = pd.read_csv(
        "output/EDA_outputs/severity_from_astronomical_twilight.csv"
    )
    civil_twilight = pd.read_csv("output/EDA_outputs/severity_from_civil_twilight.csv")
    nautical_twilight = pd.read_csv("output/EDA_outputs/severity_from_nautical_twilight.csv")
    sunrise_sunset = pd.read_csv("output/EDA_outputs/severity_from_sunrise_sunset.csv")

    st.subheader("Astronomical Twilight")
    st.markdown(
        """Begins in the morning, or ends in the evening, when the geometric center of 
        the sun is 18 degrees below the horizon. In astronomical twilight, sky illumination 
        is so faint that most casual observers would regard the sky as fully dark, 
        especially under urban or suburban light pollution."""
    )
    astronomical_twilight = (
        astronomical_twilight.groupby(["severity", "astronomical_twilight"])
        .first()
        .unstack(level=-1)
    )
    astronomical_twilight = astronomical_twilight.xs("count", axis=1).reset_index()
    chart = (
        alt.Chart(astronomical_twilight)
        .transform_fold(["Day", "Night"], as_=["key", "value"])
        .mark_bar()
        .encode(
            alt.X("key:N", axis=None),
            alt.Y("value:Q"),
            alt.Color("key:N", legend=alt.Legend(title=None, orient="bottom")),
            alt.Column(
                "severity",
                sort=astronomical_twilight["severity"].values,
                header=alt.Header(labelOrient="bottom", title=None),
            ),
        )
    )
    st.altair_chart(chart)
    st.markdown("Severity 1 and 3 happen in day")
    st.divider()

    st.subheader("Civil Twilight")
    st.markdown(
        """Begins in the morning, or ends in the evening, when the geometric center of the sun 
        is 6 degrees below the horizon. Therefore morning civil twilight begins when the 
        geometric center of the sun is 6 degrees below the horizon, and ends at sunrise."""
    )
    civil_twilight = (
        civil_twilight.groupby(["severity", "civil_twilight"]).first().unstack(level=-1)
    )
    civil_twilight = civil_twilight.xs("count", axis=1).reset_index()
    chart = (
        alt.Chart(civil_twilight)
        .transform_fold(["Day", "Night"], as_=["key", "value"])
        .mark_bar()
        .encode(
            alt.X("key:N", axis=None),
            alt.Y("value:Q"),
            alt.Color("key:N", legend=alt.Legend(title=None, orient="bottom")),
            alt.Column(
                "severity",
                sort=civil_twilight["severity"].values,
                header=alt.Header(labelOrient="bottom", title=None),
            ),
        )
    )
    st.altair_chart(chart)
    st.markdown("Severity 1 and 3 happen in day")
    st.divider()

    st.subheader("Nautical Twilight")
    st.markdown(
        """Begins in the morning, or ends in the evening, when 
        the geometric center of the sun is 12 degrees below the horizon."""
    )
    nautical_twilight = (
        nautical_twilight.groupby(["severity", "nautical_twilight"])
        .first()
        .unstack(level=-1)
    )
    nautical_twilight = nautical_twilight.xs("count", axis=1).reset_index()
    chart = (
        alt.Chart(nautical_twilight)
        .transform_fold(["Day", "Night"], as_=["key", "value"])
        .mark_bar()
        .encode(
            alt.X("key:N", axis=None),
            alt.Y("value:Q"),
            alt.Color("key:N", legend=alt.Legend(title=None, orient="bottom")),
            alt.Column(
                "severity",
                sort=nautical_twilight["severity"].values,
                header=alt.Header(labelOrient="bottom", title=None),
            ),
        )
    )
    st.altair_chart(chart)
    st.markdown("Severity 1 and 3 happen in day")
    st.divider()

    st.subheader("Sunrise Sunset")
    st.markdown("The period of day (i.e. day or night) based on sunrise/sunset.")
    sunrise_sunset = (
        sunrise_sunset.groupby(["severity", "sunrise_sunset"]).first().unstack(level=-1)
    )
    sunrise_sunset = sunrise_sunset.xs("count", axis=1).reset_index()
    chart = (
        alt.Chart(sunrise_sunset)
        .transform_fold(["Day", "Night"], as_=["key", "value"])
        .mark_bar()
        .encode(
            alt.X("key:N", axis=None),
            alt.Y("value:Q"),
            alt.Color("key:N", legend=alt.Legend(title=None, orient="bottom")),
            alt.Column(
                "severity",
                sort=sunrise_sunset["severity"].values,
                header=alt.Header(labelOrient="bottom", title=None),
            ),
        )
    )
    st.altair_chart(chart)
    st.markdown("Severity 1 and 3 happen in day")
    st.divider()


def display_weather():
    """
    Printing weather information
    """
    pressure = pd.read_csv("output/EDA_outputs/severity_from_pressure.csv")
    humidity = pd.read_csv("output/EDA_outputs/severity_from_humidity.csv")
    visibility = pd.read_csv("output/EDA_outputs/severity_from_visibility.csv")

    st.subheader("Pressure")
    pressure = pressure.set_index("severity")
    st.bar_chart(pressure)
    st.markdown("There are no correlation between pressure and target.")
    st.divider()

    st.subheader("Humidity")
    humidity = humidity.set_index("severity")
    st.bar_chart(humidity)
    st.markdown("There are no correlation between pressure and target.")
    st.divider()

    st.subheader("Visibility")
    visibility = visibility.set_index("severity")
    st.bar_chart(visibility)
    st.markdown("There are no correlation between pressure and target.")
    st.divider()


def display_poi():
    """
    Printing point of interest objects
    """
    railway = pd.read_csv("output/EDA_outputs/severity_from_railway.csv")
    roundabout = pd.read_csv("output/EDA_outputs/severity_from_roundabout.csv")
    station = pd.read_csv("output/EDA_outputs/severity_from_station.csv")

    st.subheader("Railway")
    railway = railway.groupby(["severity", "railway"]).first().unstack(level=-1)
    railway = railway.xs("count", axis=1).reset_index()
    chart = (
        alt.Chart(railway)
        .transform_fold(["True", "False"], as_=["key", "value"])
        .mark_bar()
        .encode(
            alt.X("key:N", axis=None),
            alt.Y("value:Q"),
            alt.Color("key:N", legend=alt.Legend(title=None, orient="bottom")),
            alt.Column(
                "severity",
                sort=railway["severity"].values,
                header=alt.Header(labelOrient="bottom", title=None),
            ),
        )
    )
    st.altair_chart(chart)
    st.markdown("There are only small amount of accidents near railway.")
    st.divider()

    st.subheader("Roundbout")
    roundabout = (
        roundabout.groupby(["severity", "roundabout"]).first().unstack(level=-1)
    )
    roundabout = roundabout.xs("count", axis=1).reset_index()
    chart = (
        alt.Chart(roundabout)
        .transform_fold(["True", "False"], as_=["key", "value"])
        .mark_bar()
        .encode(
            alt.X("key:N", axis=None),
            alt.Y("value:Q"),
            alt.Color("key:N", legend=alt.Legend(title=None, orient="bottom")),
            alt.Column(
                "severity",
                sort=roundabout["severity"].values,
                header=alt.Header(labelOrient="bottom", title=None),
            ),
        )
    )
    st.altair_chart(chart)
    st.divider()

    st.subheader("Station")
    station = station.groupby(["severity", "station"]).first().unstack(level=-1)
    station = station.xs("count", axis=1).reset_index()
    chart = (
        alt.Chart(station)
        .transform_fold(["True", "False"], as_=["key", "value"])
        .mark_bar()
        .encode(
            alt.X("key:N", axis=None),
            alt.Y("value:Q"),
            alt.Color("key:N", legend=alt.Legend(title=None, orient="bottom")),
            alt.Column(
                "severity",
                sort=station["severity"].values,
                header=alt.Header(labelOrient="bottom", title=None),
            ),
        )
    )
    st.altair_chart(chart)
    st.markdown("There are only small amount of accidents near stations.")
    st.divider()


def display_traffic_info():
    """
    Printing traffic signs
    """
    stop = pd.read_csv("output/EDA_outputs/severity_from_stop.csv")
    turning_loop = pd.read_csv("output/EDA_outputs/severity_from_turning_loop.csv")
    traffic_signal = pd.read_csv("output/EDA_outputs/severity_from_traffic_signal.csv")
    traffic_calming = pd.read_csv("output/EDA_outputs/severity_from_traffic_calming.csv")

    st.subheader("Stop sign")
    stop = stop.groupby(["severity", "stop"]).first().unstack(level=-1)
    stop = stop.xs("count", axis=1).reset_index()
    chart = (
        alt.Chart(stop)
        .transform_fold(["True", "False"], as_=["key", "value"])
        .mark_bar()
        .encode(
            alt.X("key:N", axis=None),
            alt.Y("value:Q"),
            alt.Color("key:N", legend=alt.Legend(title=None, orient="bottom")),
            alt.Column(
                "severity",
                sort=stop["severity"].values,
                header=alt.Header(labelOrient="bottom", title=None),
            ),
        )
    )
    st.altair_chart(chart)
    st.markdown("There are only small amount of accidents near stop sign.")
    st.divider()

    st.subheader("Turning loop")
    turning_loop = (
        turning_loop.groupby(["severity", "turning_loop"]).first().unstack(level=-1)
    )
    turning_loop = turning_loop.xs("count", axis=1).reset_index()
    chart = (
        alt.Chart(turning_loop)
        .transform_fold(["True", "False"], as_=["key", "value"])
        .mark_bar()
        .encode(
            alt.X("key:N", axis=None),
            alt.Y("value:Q"),
            alt.Color("key:N", legend=alt.Legend(title=None, orient="bottom")),
            alt.Column(
                "severity",
                sort=turning_loop["severity"].values,
                header=alt.Header(labelOrient="bottom", title=None),
            ),
        )
    )
    st.altair_chart(chart)
    st.markdown("There are only small amount of accidents near turning loop.")
    st.divider()

    st.subheader("Traffic signal")
    traffic_signal = (
        traffic_signal.groupby(["severity", "traffic_signal"]).first().unstack(level=-1)
    )
    traffic_signal = traffic_signal.xs("count", axis=1).reset_index()
    chart = (
        alt.Chart(traffic_signal)
        .transform_fold(["True", "False"], as_=["key", "value"])
        .mark_bar()
        .encode(
            alt.X("key:N", axis=None),
            alt.Y("value:Q"),
            alt.Color("key:N", legend=alt.Legend(title=None, orient="bottom")),
            alt.Column(
                "severity",
                sort=traffic_signal["severity"].values,
                header=alt.Header(labelOrient="bottom", title=None),
            ),
        )
    )
    st.altair_chart(chart)
    st.markdown(
        """There are only small amount of accidents near traffic signal. 
        In severity 1 more than half are near traffic signal."""
    )
    st.divider()

    st.subheader("Traffic Calming")
    traffic_calming = (
        traffic_calming.groupby(["severity", "traffic_calming"])
        .first()
        .unstack(level=-1)
    )
    traffic_calming = traffic_calming.xs("count", axis=1).reset_index()
    chart = (
        alt.Chart(traffic_calming)
        .transform_fold(["True", "False"], as_=["key", "value"])
        .mark_bar()
        .encode(
            alt.X("key:N", axis=None),
            alt.Y("value:Q"),
            alt.Color("key:N", legend=alt.Legend(title=None, orient="bottom")),
            alt.Column(
                "severity",
                sort=traffic_calming["severity"].values,
                header=alt.Header(labelOrient="bottom", title=None),
            ),
        )
    )
    st.altair_chart(chart)
    st.markdown("There are only small amount of accidents near traffic calming.")
    st.divider()


def display_pda():
    """
    Printning pda information
    """

    st.header("PDA")
    st.subheader('Final metrics')
    results = pd.DataFrame(
        [
            ["DecisionTreeClassifier", "0.7231424779214289", "0.773183146357127"],
            ["RandomForestClassifier", "0.6662014750820551", " 0.7682111899384734"],
        ],
        columns=["Model", "Default", "Finetuned"],
    )
    st.table(results)

    decision_tree_res = pd.DataFrame(
        [
            (0.7041790974048232, 5, 16),
            (0.7108394087000414, 5, 32),
            (0.7145315264501846, 5, 64),
            (0.7292767500943312, 7, 16),
            (0.7346144417852223, 7, 32),
            (0.7379026626855651, 7, 64),
            (0.7668999438697521, 10, 16),
            (0.7746654964665024, 10, 32),
            (0.7808324461108587, 10, 64),
        ],
        columns=["f1", " maxDepth", "maxBins"],
    )

    random_forest_res = pd.DataFrame(
        [
            (0.6601881957161798, 10, 5),
            (0.7155148650244513, 10, 7),
            (0.7518180686454068, 10, 10),
            (0.6735103333231615, 20, 5),
            (0.7236128701958222, 20, 7),
            (0.7588463808238362, 20, 10),
            (0.6777265636757628, 30, 5),
            (0.7285121458610525, 30, 7),
            (0.7641123600346675, 30, 10),
        ],
        columns=["f1", "numTrees", "maxDepth"],
    )

    st.subheader("Hyperparameters tuning: Decision Tree")
    st.table(decision_tree_res)
    st.subheader("Hyperparameters tuning: Random Forest")
    st.table(random_forest_res[["numTrees", "maxDepth", "f1"]])
    st.divider()

    st.subheader('Output example')
    st.markdown('Decision Tree')
    output = pd.read_csv('output/DT_predictions.csv')
    st.table(output.sample(10).head(10))
    st.markdown('Random Forest')
    output = pd.read_csv('output/RF_predictions.csv')
    st.table(output.sample(10).head(10))



if __name__ == "__main__":
    main()
