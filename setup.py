from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["astropy>=4", "numpy>=1.18", "scipy>=1.4", "matplotlib>=3.1"]

setup(name="pmoired",
      version="0.1dev",
      author="Antoine Merand",
      author_email="amerand@eso.org",
      description="Read/Display/Fit astronomical interferometric data in OIFITS format",
      long_description = readme,
      long_description_content_type="text/markdown",
      url="https://github.com/amerand/PMOIRED",
      packages=find_packages(),
      classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Simplified BSD License",
        "Operating System :: OS Independent",
      ],
      license="LICENSE.txt",
      include_package_data=True,
      python_requires=">=3.7"
)
