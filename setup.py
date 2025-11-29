import os
from setuptools import setup, find_packages

VERSION = "0.0.1"

def get_requirements():
    try:
        with open("./requirements.txt") as reqsf:
            reqs = [line.strip() for line in reqsf.readlines() if line.strip()]
    except FileNotFoundError:
        reqs = []
    return reqs

if __name__ == "__main__":
    setup(
        name="kkospi",
        version=VERSION,
        python_requires=">=3.9",
        install_requires=get_requirements(),
        
        packages=find_packages(
            include=['model*', 'data*', 'trainers*', 'config*', 'utils*'],
            exclude=['runs', 'result', 'checkpoints', 'tests', 'docs']
        ),
    )