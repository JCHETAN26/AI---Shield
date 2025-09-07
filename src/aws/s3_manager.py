"""
S3 Manager for handling AWS S3 operations in AI Shield.

This module provides functionality to:
- Connect to S3 buckets
- Download models and datasets
- Upload results and logs
- Handle S3 authentication and error handling
"""

import boto3
import logging
from pathlib import Path
from typing import Optional
from botocore.exceptions import ClientError, NoCredentialsError


class S3Manager:
    """
    Manages S3 operations for the AI Shield project.
    
    Handles downloading models/data from S3 and uploading results.
    """
    
    def __init__(self, bucket_name: str, region: str = "us-east-1"):
        """
        Initialize S3 manager.
        
        Args:
            bucket_name: Name of the S3 bucket
            region: AWS region
        """
        self.bucket_name = bucket_name
        self.region = region
        self.logger = logging.getLogger(__name__)
        
        try:
            self.s3_client = boto3.client('s3', region_name=region)
            self._verify_bucket_access()
            self.logger.info(f"S3Manager initialized for bucket: {bucket_name}")
        except NoCredentialsError:
            self.logger.error("AWS credentials not found. Please configure AWS credentials.")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing S3 client: {str(e)}")
            raise
    
    def _verify_bucket_access(self):
        """Verify that the bucket exists and is accessible."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise ValueError(f"S3 bucket '{self.bucket_name}' does not exist")
            elif error_code == '403':
                raise ValueError(f"Access denied to S3 bucket '{self.bucket_name}'")
            else:
                raise
    
    def download_file(self, s3_key: str, local_path: str) -> str:
        """
        Download a file from S3 to local storage.
        
        Args:
            s3_key: S3 object key (path in bucket)
            local_path: Local file path to save the downloaded file
            
        Returns:
            Local file path where the file was saved
        """
        try:
            # Ensure local directory exists
            local_file_path = Path(local_path)
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Downloading {s3_key} from bucket {self.bucket_name}")
            
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                str(local_file_path)
            )
            
            self.logger.info(f"File downloaded successfully to {local_file_path}")
            return str(local_file_path)
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise FileNotFoundError(f"File '{s3_key}' not found in bucket '{self.bucket_name}'")
            else:
                self.logger.error(f"Error downloading file: {str(e)}")
                raise
        except Exception as e:
            self.logger.error(f"Unexpected error downloading file: {str(e)}")
            raise
    
    def upload_file(self, local_path: str, s3_key: str) -> bool:
        """
        Upload a local file to S3.
        
        Args:
            local_path: Path to the local file
            s3_key: S3 object key (destination path in bucket)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Uploading {local_path} to {s3_key}")
            
            self.s3_client.upload_file(
                local_path,
                self.bucket_name,
                s3_key
            )
            
            self.logger.info(f"File uploaded successfully to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except FileNotFoundError:
            self.logger.error(f"Local file not found: {local_path}")
            return False
        except Exception as e:
            self.logger.error(f"Error uploading file: {str(e)}")
            return False
    
    def list_objects(self, prefix: str = "") -> list:
        """
        List objects in the S3 bucket with optional prefix filter.
        
        Args:
            prefix: Object key prefix to filter by
            
        Returns:
            List of object keys
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error listing objects: {str(e)}")
            return []
    
    def object_exists(self, s3_key: str) -> bool:
        """
        Check if an object exists in the S3 bucket.
        
        Args:
            s3_key: S3 object key to check
            
        Returns:
            True if object exists, False otherwise
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                raise
    
    def get_object_size(self, s3_key: str) -> Optional[int]:
        """
        Get the size of an S3 object in bytes.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Size in bytes, or None if object doesn't exist
        """
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return response['ContentLength']
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return None
            else:
                raise
    
    def create_presigned_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate a presigned URL for an S3 object.
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds
            
        Returns:
            Presigned URL string, or None if error
        """
        try:
            response = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return response
        except Exception as e:
            self.logger.error(f"Error generating presigned URL: {str(e)}")
            return None