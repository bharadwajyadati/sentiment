"""
Download the pretraind model from s3

Training is done using google colab or in amazon with gpu instances and
the model is saved in s3 (aviso)

"""

import os
import json
import boto3
import tarfile
import logging
import requests


__version__ = 0.1

logger = logging.getLogger(__name__)

"""
    Generic utils required for loading configuration files, along with
    download/extract data .

"""

BOTO_CLIENT = "s3"  # storing model file in s3


class GenericUtils(object):

    """
        function to load configuration from json file.
        :param config_file: json config file to load the data

    """

    @staticmethod
    def load_config(config_file):
        try:
            with open(config_file) as json_file:
                config = json.load(json_file)
            return config
        except ValueError:
            raise Exception

    """
        Download the tar file from url and extract it present location.
        Mainly used to download training data to local

        :param url: download data from url.
        :param file_name: file name after extraction.

    """

    @staticmethod
    def download_and_extract(url, file_name):
        try:
            resp = requests.get(url)
            zname = os.path.join('.', file_name)
            logger.info("Downloading file..... ")
            with open(zname, 'wb') as fp:
                fp.write(resp.content)
            logger.info("Extracting file ..... ")
            tf = tarfile.open(file_name)
            tf.extractall()
            logger.info("File extracted to current location")
        except Exception as exp:
            logger.error("error while downloading/extracting" + exp)
            raise Exception("error while downloading/extracting" + exp)

    """
        Download trained model from s3 to local

        :param bucket_name: s3 bucketname.
        :param file_name: file locaation in s3 bucket
        :param local_file: to download locally as file.

    """

    @staticmethod
    def s3downloader(bucket_name, file_name, local_file):
        try:
            s3 = boto3.client(BOTO_CLIENT, aws_access_key_id="",
                              aws_secret_access_key="")
            s3.download_file(bucket_name, file_name, local_file)
        except Exception as exp:
            logger.error("downloading from s3 failed" + exp)
            raise Exception("downloading from s3 failed" + exp)
