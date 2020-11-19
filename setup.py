import setuptools


setuptools.setup(
    name="graph-walker",
    version="0.0.1",
    author="Maixent Chenebaux",
    author_email="max.chbx@gmail.com",
    description="Fast random walks on graph",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=["networkx", "numpy", "scipy", "numba"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5"
)