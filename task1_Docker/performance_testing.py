# Performance Testing for Pandas, Polars, and PySpark
import pandas as pd
import polars as pl
from pyspark.sql import SparkSession
import time

# Data files
hom = 'homogeneous_data.csv'
het = 'fl_1.csv'

# Initialize SparkSession for PySpark
spark = SparkSession.builder.appName("PerformanceTesting").getOrCreate()

# Helper functions for each framework
def test_load_files_pandas(homogeneous, heterogeneous):
    start_time = time.time()
    df_hom = pd.read_csv(homogeneous)
    hom_time = time.time() - start_time

    start_time = time.time()
    df_het = pd.read_csv(heterogeneous)
    het_time = time.time() - start_time

    return df_hom, df_het, hom_time, het_time


def test_load_files_polars(homogeneous, heterogeneous):
    start_time = time.time()
    df_hom = pl.read_csv(homogeneous)
    hom_time = time.time() - start_time

    start_time = time.time()
    df_het = pl.read_csv(heterogeneous)
    het_time = time.time() - start_time

    return df_hom, df_het, hom_time, het_time


def test_load_files_pyspark(homogeneous, heterogeneous):
    start_time = time.time()
    df_hom = spark.read.csv(homogeneous, header=True, inferSchema=True)
    hom_time = time.time() - start_time

    start_time = time.time()
    df_het = spark.read.csv(heterogeneous, header=True, inferSchema=True)
    het_time = time.time() - start_time

    return df_hom, df_het, hom_time, het_time


def test_profiling_speed_pandas(hom, het):
    start_time = time.time()
    city_counts = het.groupby("City").size().reset_index(name="Count of People in each City").sort_values(
        by="Count of People in each City", ascending=False
    )
    het_time = time.time() - start_time

    start_time = time.time()
    counts = hom.groupby("col_0").size().reset_index(name="Count").sort_values(by="Count", ascending=False)
    hom_time = time.time() - start_time

    return counts, city_counts, hom_time, het_time


def test_profiling_speed_polars(hom, het):
    # Profiling for the heterogeneous dataset
    start_time = time.time()
    city_counts = (
        het.group_by("City")
        .agg(pl.col("City").count().alias("Count of People in each City"))
        .sort("Count of People in each City", descending=True)  # Corrected keyword argument
    )
    het_time = time.time() - start_time

    # Profiling for the homogeneous dataset
    start_time = time.time()
    counts = (
        hom.group_by("col_0")
        .agg(pl.col("col_0").count().alias("Count"))
        .sort("Count", descending=True)  # Corrected keyword argument
    )
    hom_time = time.time() - start_time

    return counts, city_counts, hom_time, het_time



def test_profiling_speed_pyspark(hom, het):
    start_time = time.time()
    city_counts = het.groupBy("City").count().orderBy("count", ascending=False).toPandas()
    het_time = time.time() - start_time

    start_time = time.time()
    counts = hom.groupBy("col_0").count().orderBy("count", ascending=False).toPandas()
    hom_time = time.time() - start_time

    return counts, city_counts, hom_time, het_time


# Test loading and profiling with Pandas
df_hom_pd, df_het_pd, hom_load_time_pd, het_load_time_pd = test_load_files_pandas(hom, het)
sampled_het_pd = df_het_pd.sample(n=10_000, random_state=42).iloc[:, :9]
hom_counts_pd, het_counts_pd, hom_prof_time_pd, het_prof_time_pd = test_profiling_speed_pandas(df_hom_pd, sampled_het_pd)

# Test loading and profiling with Polars
df_hom_pl, df_het_pl, hom_load_time_pl, het_load_time_pl = test_load_files_polars(hom, het)
sampled_het_pl = df_het_pl.sample(n=10_000).select(df_het_pl.columns[:9])
hom_counts_pl, het_counts_pl, hom_prof_time_pl, het_prof_time_pl = test_profiling_speed_polars(df_hom_pl, sampled_het_pl)

# Test loading and profiling with PySpark
df_hom_ps, df_het_ps, hom_load_time_ps, het_load_time_ps = test_load_files_pyspark(hom, het)
sampled_het_ps = df_het_ps.sample(withReplacement=False, fraction=0.001).select(df_het_ps.columns[:9])
hom_counts_ps, het_counts_ps, hom_prof_time_ps, het_prof_time_ps = test_profiling_speed_pyspark(df_hom_ps, sampled_het_ps)

# Log results to a CSV
results = pd.DataFrame({
    "Task": [
        "Load Homogeneous Data",
        "Load Heterogeneous Data",
        "Profile Homogeneous Data",
        "Profile Heterogeneous Data"
    ],
    "Pandas Time (seconds)": [
        hom_load_time_pd,
        het_load_time_pd,
        hom_prof_time_pd,
        het_prof_time_pd
    ],
    "Polars Time (seconds)": [
        hom_load_time_pl,
        het_load_time_pl,
        hom_prof_time_pl,
        het_prof_time_pl
    ],
    "PySpark Time (seconds)": [
        hom_load_time_ps,
        het_load_time_ps,
        hom_prof_time_ps,
        het_prof_time_ps
    ]
})

# Save results to a CSV file
results.to_csv("performance_results.csv", index=False)

print("Performance results saved to performance_results.csv")
