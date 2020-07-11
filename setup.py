"""
    Setup file for creating required packages for sentiment analysis
"""

from setuptools import setup, find_packages

extras = {}

extras["sklearn"] = ["scikit-learn"]  # for all the ml metrics

# TODO: finalize and set up
#extras["serving"] = ["pydantic", "uvicorn", "fastapi", "starlette"]

extras["testing"] = ["pytest", "pytest-xdist", "pyunit"]  # for unit testing

extras["docs"] = ["recommonmark", "sphinx", "sphinx-markdown-tables",
                  "sphinx-rtd-theme"]  # for sphnix documentation


setup(
    name="nlptools",
    version="0.0.1",
    author="bharadwaj",
    description="NLP tools for transcription including tf-idf , sentiment, ner, topic modelling etc .",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "numpy",
        # default framework as of now for entire NLP tasks
        "torch",
        # for reading datasets and for training
        "torchtext",
        # default library for bert models and other NLP models
        "transformers",
        # pytorch-lighting for heavy lifting of pytorch code
        "pytorch-lightning",
        # handling json
        "jsonschema",
        # progress bars in model download and training scripts
        "tqdm",
        # for downloading models over HTTPS
        "requests",
        # download and upload data to s3
        "boto3",
        "allennlp",
        "allennlp-models"
    ],
    extras_require=extras,
    python_requires=">=3.6.0",

)
