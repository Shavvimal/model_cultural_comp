import aioboto3
import aiofiles
from tqdm.asyncio import tqdm_asyncio
from pathlib import Path


class S3Download:
    def __init__(self):
        self.region = 'eu-west-2'
        self.bucket_name = 'dcypher-podcasts'

    async def list_files_in_bucket(self, prefix):
        """
        Asynchronously list all files in an S3 bucket with the given prefix.
        :param prefix: Prefix to filter the files
        :return: List of file keys in the S3 bucket with the given prefix.
        """
        session = aioboto3.Session(
            region_name=self.region,
        )
        async with session.client('s3') as s3_client:
            response = await s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            file_keys = [content['Key'] for content in response.get('Contents', [])]
            # If there are no files, print
            if not file_keys:
                print(f"\033[91mNo files found in {prefix}\033[0m")
            return file_keys

    async def get_s3_file(self, file_key):
        """
        Asynchronously fetch a file from S3 and saves it locally.
        """
        # Replace : with - in file key
        file_key_replaced = file_key.replace(":", "-")
        # Make sure the directory exists
        local_file_path = Path(f'data/{file_key_replaced}')
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        session = aioboto3.Session(region_name=self.region)

        async with session.client('s3') as s3_client:
            response = await s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            async with response['Body'] as stream:
                data = await stream.read()

            async with aiofiles.open(local_file_path, 'wb') as f:
                await f.write(data)

    async def download_all_data(self, prefix: str):
        """
        Asynchronously download all files in the S3 bucket with the given prefix.
        """
        file_keys = await self.list_files_in_bucket(prefix)
        print(f"Downloading {len(file_keys)} files\n\n")
        await tqdm_asyncio.gather(*[self.get_s3_file(file_key) for file_key in file_keys])


if __name__ == '__main__':
    import asyncio


    async def main():
        s3_downloader = S3Download()
        prefix = "2024-05-03 21.04.55 Alexander Girardet's Zoom Meeting"
        await s3_downloader.download_all_data(prefix)


    asyncio.run(main())
