from setuptools import find_namespace_packages, find_packages, setup

base_packages = find_packages(where="src")
namespace_packages = find_namespace_packages(where="src")
packages = base_packages + [pkg for pkg in namespace_packages if pkg not in base_packages]

setup(
    name="ai-patterns-mining",
    version="0.1.0",
    author="Your Name",
    author_email="hasinthakapiyumal@gmail.com",
    description="AI Pattern Mining Pipeline",
    packages=packages,
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "langchain",
        "langchain-text-splitters",
        "langchain[google-genai]",
        "pypdf",
        "requests",
        "numpy",
        "pandas",
        "scikit-learn",
        "sentence-transformers",
        "PyMuPDF",
        "torch",
        "tqdm",
        "pyyaml",
        "joblib",
        "matplotlib",
        "seaborn",
        "umap-learn",
    ],
    entry_points={
        "console_scripts": [
            "ai-patterns-mining=pipeline.main:main",
        ],
    },
    python_requires=">=3.9",
)
