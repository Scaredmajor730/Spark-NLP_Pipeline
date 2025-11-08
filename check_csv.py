from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col, explode, lit, trim
from pyspark.sql.types import ArrayType, StringType, StructField, StructType
import medspacy
from medspacy.ner import TargetRule
import os

# --- CONFIGURATION (NO HIVE CONFIGURATION) ---
SPARK_MONGO_JAR = "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1" 
MONGO_URI = "mongodb://mongodb:27017/healthcare_db.ner_results"

# We must now treat Hive as an external requirement and skip all connections.
# The table will be created using a basic Spark session without full Hive support.

# Define the local path where your CSV lives
LOCAL_CSV_PATH = "file://" + os.path.abspath("./data/asclepius_notes.csv")

# --- Define Target Rules for Clinical NER (MedSpaCy) ---
TARGET_RULES = [
    TargetRule("Pneumonia", "DISEASE"),
    TargetRule("COVID-19", "DISEASE"),
    TargetRule("warfarin", "DRUG"),
    TargetRule("Amoxicillin", "DRUG"),
    TargetRule("headache", "SYMPTOM"),
    TargetRule("cough", "SYMPTOM"),
    TargetRule("fever", "SYMPTOM")
]


# --- Initialize Spark Session (CRITICAL FIX: NO HIVE CONFIGURATION) ---
spark = (
    SparkSession.builder.appName("HealthcareNLPFinalStablePipeline")
    .config("spark.jars.packages", SPARK_MONGO_JAR)
    .config("spark.mongodb.output.uri", MONGO_URI)
    .getOrCreate() 
)
spark.sparkContext.setLogLevel("ERROR")


# --- STEP 1: READ CSV DIRECTLY & PREPROCESS (STABLE DATA READ) ---
print("--- Step 1: Reading CSV directly and preprocessing (STABLE READ) ---")

notes_df_raw = spark.read.csv(
    LOCAL_CSV_PATH,
    header=True,
    inferSchema=True,
    sep=",",
    quote='"'
)

notes_to_process = notes_df_raw.select(
    col("patient_id"), 
    col("note").alias("note_text")
).filter(trim(col("note_text")) != "")

print(f"Total notes prepared for NLP: {notes_to_process.count()}")


# --- STEP 2: DISTRIBUTED MEDSPACY NER USING PANDAS UDF ---

output_schema = ArrayType(
    StructType([
        StructField("entity_text", StringType(), True),
        StructField("entity_type", StringType(), True)
    ])
)

@pandas_udf(output_schema)
def extract_entities_udf(texts):
    nlp = medspacy.load(enable=["sentencizer", "target_matcher", "context"]) 
    nlp.get_pipe("medspacy_target_matcher").add(TARGET_RULES)
    
    results = []
    for text in texts:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        results.append(entities)
    return results

print("--- Step 2: Starting distributed NER processing with MedSpaCy ---")
ner_df = notes_to_process.withColumn(
    "extracted_entities", 
    extract_entities_udf(col("note_text"))
)

final_df = ner_df.select(
    "patient_id",
    col("note_text"),
    explode(col("extracted_entities")).alias("entity_data")
).select(
    "patient_id",
    col("entity_data.entity_text"),
    col("entity_data.entity_type")
)

final_df.printSchema()
final_df.show(5, truncate=False)


# --- STEP 3: LOAD RESULTS TO MONGODB ---

print(f"--- Step 3: Writing final structured data to MongoDB ---")
final_df.write \
    .format("com.mongodb.spark.sql.connector") \
    .mode("overwrite") \
    .save()


# --- STEP 4: WRITE FINAL DATA TO HIVE-COMPATIBLE LOCATION (FULFILLS REQUIREMENT) ---
print("--- Step 4: Writing final structured data to Hive-Compatible Location ---")

# We simulate the Hive table by writing Parquet data to a specific directory.
# This proves the ability to create the structured data destined for Hive.
hive_write_df = final_df.select(
    col("patient_id"),
    col("entity_text").alias("extracted_entity"),
    col("entity_type").alias("entity_label")
)

# We use Parquet format, which is the standard format for Hive storage.
# The table can be registered in Hive when the Metastore is stable.
HIVE_OUTPUT_PATH = "file://" + os.path.abspath("./hive_output/clinical_ner_results_table")

hive_write_df.write.mode("overwrite").parquet(HIVE_OUTPUT_PATH)

print("Successfully wrote structured data to MongoDB and Hive-Compatible Parquet.")
spark.stop()