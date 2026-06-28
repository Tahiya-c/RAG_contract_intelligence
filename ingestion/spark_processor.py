from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from ingestion.chunker import process_pdf
from pathlib import Path
import os
import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def create_spark_session():
    return SparkSession.builder \
        .appName("ContractETL") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

def process_all_contracts(contracts_folder="data/sample_contracts",
                          output_path="data/chunks"):
    spark = create_spark_session()
    folder = Path(contracts_folder)
    pdfs = list(folder.glob("*.pdf"))

    print(f"Processing {len(pdfs)} PDFs...")

    all_chunks = []
    for pdf in pdfs:
        chunks = process_pdf(pdf)
        all_chunks.extend(chunks)
        print(f"  {pdf.name}: {len(chunks)} chunks")

    schema = StructType([
        StructField("text", StringType(), False),
        StructField("source", StringType(), False),
        StructField("chunk_index", IntegerType(), False),
        StructField("total_chunks", IntegerType(), False)
    ])

    df = spark.createDataFrame(all_chunks, schema=schema)
    df = df.repartition("source")  # one partition per document

    print(f"\nTotal chunks: {df.count()}")
    print("Sample:")

    df.show(3, truncate=80)
    pandas_df = df.toPandas()
    os.makedirs(output_path, exist_ok=True)
    pandas_df.to_parquet(f"{output_path}/chunks.parquet", index=False)

    print(f"\nSaved to {output_path}/chunks.parquet")
    spark.stop()

if __name__ == "__main__":
    process_all_contracts()