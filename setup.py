from setuptools import setup, find_packages

setup(
    name="gpudrive",
    version="0.1.0",
    packages=find_packages(include=["pygpudrive"]),
    package_data={
        "gpudrive": ["cpython-31*-*.so"],
        "pygpudrive"
    },
    include_package_data=True,
)
