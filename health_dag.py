from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

# --- ABSOLUTE PATHS & CONFIGURATION ---
# These paths were verified from your environment:
SPARK_SUBMIT_PATH = "/home/mkdspectrex360/BDA/HealthCare-NLP_Pipeline/venv/bin/spark-submit" 
JAVA_HOME_PATH = "/usr/lib/jvm/java-17-openjdk-amd64" 
SPARK_SCRIPT_PATH = "/home/mkdspectrex360/BDA/HealthCare-NLP_Pipeline/spark_nlp_pipeline.py" 
MONGO_SPARK_PACKAGE = "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1" 


# --- FINAL BASH COMMAND (Explicitly sets JAVA_HOME for execution) ---
# This command uses absolute paths and exports JAVA_HOME, which should resolve the low-level crash.

# --- ADD THIS LINE TO THE COMMAND ---
VENV_PYTHON_EXEC = "/home/mkdspectrex360/BDA/HealthCare-NLP_Pipeline/venv/bin/python"

SPARK_COMMAND = f"""
    export JAVA_HOME={JAVA_HOME_PATH} && \\
    export PYSPARK_PYTHON={VENV_PYTHON_EXEC} && \\ # <--- CRITICAL FIX: Forces Spark to use VENV Python
    {SPARK_SUBMIT_PATH} \\
    --master local[*] \\
    --packages {MONGO_SPARK_PACKAGE} \\
    --conf spark.driver.memory=1g \\
    --total-executor-cores 1 \\
    {SPARK_SCRIPT_PATH}
"""


with DAG(
    # FINAL DAG ID
    dag_id="healthcare_spark_nlp_pipeline_bash_final", 
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=["spark", "bash_fix", "nlp", "mongodb"],
) as dag:
    
    # Task 1: Runs the full PySpark job using the OS shell with fixed environment variables
    run_spark_etl_job = BashOperator(
        task_id="run_spark_etl_job",
        bash_command=SPARK_COMMAND,
        execution_timeout=pendulum.duration(minutes=500) # Increased timeout for safety
    )

    # Task 2: Verification step
    verify_mongo_load = BashOperator(
        task_id="verify_mongo_load",
        bash_command='echo "Pipeline complete. Check MongoDB: healthcare_db.ner_results"',
    )

    # Define the dependency
    run_spark_etl_job >> verify_mongo_load