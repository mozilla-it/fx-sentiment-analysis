
from setuptools import setup, find_packages

setup(name='fx-sentiment-analysis',
      version='0.0.1',
      description='airflow script to analyze fx sentiment data',
      python_requires='>=3.4',
      author='Adam Frank',
      author_email='afrank@mozilla.com',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      scripts=['data_processing.py'],
      install_requires=[
        'google-cloud-bigquery',
        'google-cloud-storage',
        'google-cloud-translate',
        'google-cloud-language',
        'langdetect',
        'numpy',
        'nltk',
        'pandas',
        'scipy',
        'sklearn',
        'selenium',
      ]
    )
