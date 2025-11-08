from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col, explode, trim, current_timestamp
from pyspark.sql.types import ArrayType, StringType, StructField, StructType
from pyspark import SparkContext # <-- NEW IMPORT
import spacy 
import pandas as pd
from pymongo import MongoClient

# --- CONFIGURATION ---
MONGO_URI = "mongodb://localhost:27017/healthcare_db.ner_results"
INPUT_FILE_PATH = "/home/mkdspectrex360/BDA/HealthCare-NLP_Pipeline/data/asclepius_notes.csv"
SPARK_MONGO_JAR = "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1" 


# --- Initialize Spark Session (Bypassing SparkSession.builder for stability) ---

# 1. Get the existing SparkContext (set up via spark-submit)
sc = SparkContext.getOrCreate()

# 2. Ensure packages are configured for the new session (This is necessary because 
# we are not using the standard .config chain that caused the error)
# We use the SparkContext's configuration object (sc._conf)
sc._conf.set("spark.jars.packages", SPARK_MONGO_JAR)
sc._conf.set("spark.mongodb.output.uri", MONGO_URI)
sc._conf.set("spark.app.name", "HealthcareNLPSimplifiedPipeline")

# 3. Create the SparkSession from the existing context
spark = SparkSession(sc) 

spark.sparkContext.setLogLevel("ERROR")


# --- STEP 1: READ DATA FROM LOCAL CSV FILE (FIXED COLUMN MAPPING) ---
print(f"--- Reading data from local CSV file: {INPUT_FILE_PATH} ---")

# FIX: Column mapping using the confirmed header: patient_id, note
notes_to_process = (
    spark.read.csv(INPUT_FILE_PATH, header=True, inferSchema=True)
    .select(
        col("patient_id"), 
        col("note").alias("note_text")
    )
    .filter(trim(col("note_text")) != "")
)

print(f"Total notes prepared for NLP: {notes_to_process.count()}")


# --- STEP 2: DISTRIBUTED SPACY NER USING PANDAS UDF ---

output_schema = ArrayType(
    StructType([
        StructField("entity_text", StringType(), True),
        StructField("entity_type", StringType(), True)
    ])
)

@pandas_udf(output_schema)
def extract_entities_udf(texts: pd.Series) -> pd.Series:
    # Load basic English spaCy model (robust and stable)
    nlp = spacy.load("en_core_web_sm") 
    
    all_results = []
    for text in texts:
        doc = nlp(str(text))
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        all_results.append(entities)
        
    return pd.Series(all_results)


print("--- Starting distributed NER processing with simplified spaCy ---")
ner_df = notes_to_process.withColumn(
    "extracted_entities", 
    extract_entities_udf(col("note_text"))
)

# Transform the results into the final structure
final_df = ner_df.select(
    "patient_id",
    col("note_text"),
    explode(col("extracted_entities")).alias("entity_data")
).select(
    "patient_id",
    col("entity_data.entity_text"),
    col("entity_data.entity_type")
).withColumn("pipeline_timestamp", current_timestamp()) 


# --- STEP 3: LOAD RESULTS TO MONGODB ---

print(f"--- Writing final structured data to MongoDB ---")
(
    final_df.write
    .format("mongo") # Using the stable format name
    .mode("overwrite")
    .option("uri", MONGO_URI) 
    .save()
)

spark.stop()
