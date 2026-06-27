import statistics
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def run_experiment():
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("aqe-partitions-exploration")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    ratings = spark.read.parquet("data/output/clean/ratings").cache()
    ratings.count()

    partitions_values = [2, 8, 16, 32]
    results = []

    for partitions in partitions_values:
        spark.conf.set("spark.sql.shuffle.partitions", str(partitions))
        durations = []

        for _ in range(3):
            start = time.perf_counter()
            query = (
                ratings.groupBy("movieId")
                .agg(F.count("rating").alias("votes"))
                .orderBy(F.desc("votes"))
            )
            query.count()
            durations.append(time.perf_counter() - start)

        avg_duration = statistics.mean(durations)
        results.append((partitions, avg_duration))
        print(f"partitions={partitions} avg_time={avg_duration:.3f}s")

    print("\nSummary:")
    for partitions, avg_duration in results:
        print(f"- partitions={partitions}: {avg_duration:.3f}s")

    spark.stop()


if __name__ == "__main__":
    run_experiment()
