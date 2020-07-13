import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="OptimalCluster",
    version="0.1.0",
    author="Shreyas Kera",
    author_email="shreykera7@gmail.com",
    description="Python implementation of various algorithms to find the optimal number of clusters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shreyas-bk/OptimalCluster",
    packages=['OptimalCluster'],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
