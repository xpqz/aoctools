import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aoctools",
    version="0.0.1",
    author="Stefan Kruger",
    author_email="stefan.kruger@gmail.com",
    description="Utility classes and algorithms for AoC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xpqz/aoctools",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    entry_points={'console_scripts': []},
    install_requires=[]
)
