from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="pyg-ssl",
    version="0.1.0",
    description="A PyTorch Geometric-based package for self-supervised learning on graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="iDEA-iSAIL LAB @ UIUC",
    author_email="violet24k@outlook.com",
    url="https://github.com/iDEA-iSAIL-Lab-UIUC/pyg-ssl",
    packages=find_packages(where="src"),  # Ensure packages are found in the src directory
    package_dir={"": "src"},  # Map the source directory to the package root
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch==2.3.1",
        "torch_geometric==2.6.1",
        "faiss-gpu==1.7.2",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "yacs",
    ],
    dependency_links=[
        "https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html#egg=dgl"
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8",
            "black",
        ],
    },
    entry_points={
        "console_scripts": [
            "pyg-ssl-cli=pyg_ssl.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
