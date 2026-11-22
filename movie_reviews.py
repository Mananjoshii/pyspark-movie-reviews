from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, trim
import os

# Create Spark Session
spark = SparkSession.builder \
    .appName("IMDb Movie Reviews Analysis") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Load Dataset
print("\n========== 1. LOADING DATASET ==========")

df = spark.read.csv(
    "IMDB Dataset.csv",
    header=True,
    inferSchema=True
)

# Screenshot 1: Dataset Schema
print("\n========== 2. DATASET SCHEMA ==========")
df.printSchema()

# Screenshot 2: First Five Reviews
print("\n========== 3. FIRST FIVE REVIEWS ==========")
df.show(5, truncate=False)

# Clean Dataset
cleaned_df = df.filter(
    col("review").isNotNull() &
    (trim(col("review")) != "")
)

# Screenshot 3: Total Reviews
print("\n========== 4. TOTAL REVIEWS ==========")

total_reviews = cleaned_df.count()

print("Total Reviews:", total_reviews)

# Screenshot 4: Positive and Negative Counts
print("\n========== 5. SENTIMENT COUNTS ==========")

cleaned_df.groupBy("sentiment") \
    .count() \
    .orderBy("sentiment") \
    .show()

# Add Review Length
transformed_df = cleaned_df.withColumn(
    "review_length",
    length(col("review"))
)

# Screenshot 5: Reviews Longer Than 500 Characters
print("\n========== 6. REVIEWS LONGER THAN 500 CHARACTERS ==========")

long_reviews = transformed_df.filter(
    col("review_length") > 500
)

long_reviews.select(
    "review",
    "sentiment",
    "review_length"
).show(10, truncate=False)

# Save Processed Dataset
print("\n========== 7. SAVING PROCESSED DATA ==========")

output_path = "processed_reviews"

transformed_df.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)

print("Processed dataset saved successfully!")
print("Output Location:", os.path.abspath(output_path))

# Screenshot 6: Output Folder Contents
print("\n========== 8. OUTPUT FOLDER CONTENTS ==========")

for file in os.listdir(output_path):
    print(file)

print("\n========== PROCESSING COMPLETED ==========")

spark.stop()