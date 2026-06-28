import boto3
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def get_minio_client():
    return boto3.client(
        "s3",
        endpoint_url=f"http://{os.getenv('MINIO_ENDPOINT')}",
        aws_access_key_id=os.getenv("MINIO_ROOT_USER"),
        aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD")
    )

def upload_contracts(folder_path, bucket_name="contracts"):
    client = get_minio_client()
    folder = Path(folder_path)
    pdfs = list(folder.glob("*.pdf"))
    
    if not pdfs:
        print(f"No PDFs found in {folder_path}")
        return
    
    for pdf in pdfs:
        client.upload_file(
            Filename=str(pdf),
            Bucket=bucket_name,
            Key=pdf.name
        )
        print(f"Uploaded: {pdf.name}")
    
    print(f"\nDone. {len(pdfs)} files uploaded to bucket '{bucket_name}'")

if __name__ == "__main__":
    upload_contracts("data/sample_contracts")