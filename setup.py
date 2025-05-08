# setup.py

import io
import os
from setuptools import setup, find_packages

# read the contents of your requirements.txt
here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='rmsearch',
    version='0.1.0',
    description='search engine by reward model',
    #long_description=io.open(os.path.join(here, 'README.md'), encoding='utf-8').read(),
    #long_description_content_type='text/markdown',
    author='Kentaro Seki',
    author_email='seki.kentaro@kyotoai.org',
    url='https://github.com/kyotoai/RMSearch',
    packages=find_packages(exclude=('tests',)),
    install_requires=requirements,
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'my-tool = my_package.cli:main',
        ],
    },
)
