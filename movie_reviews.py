from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length


spark = SparkSession.builder \
    .appName("IMDb Movie Reviews Analysis") \
    .getOrCreate()


df = spark.read.csv(
    "IMDB Dataset.csv",
    header=True,
    inferSchema=True
)


print("Dataset Schema:")
df.printSchema()

print("First 5 Reviews:")
df.show(5, truncate=False)


cleaned_df = df.filter(
    col("review").isNotNull() &
    (col("review") != "")
)


print("Total Reviews:", cleaned_df.count())


cleaned_df.groupBy("sentiment").count().show()


transformed_df = cleaned_df.withColumn(
    "review_length",
    length(col("review"))
)


print("Reviews longer than 500 characters:")
transformed_df.filter(
    col("review_length") > 500
).show(10, truncate=False)


transformed_df.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("processed_reviews")

print("Dataset processing completed successfully!")

spark.stop()