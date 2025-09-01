# File location: setup.py

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lightning-master-pro",
    version="0.1.0",
    author="LightningMasterPro Team",
    author_email="contact@lightningmasterpro.dev",
    description="A comprehensive PyTorch Lightning learning framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/LightningMasterPro",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "export": [
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lmpro=lmpro.cli:main",
            "lmpro-train=scripts.train:main",
            "lmpro-predict=scripts.predict:main",
            "lmpro-export=scripts.export_onnx:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/LightningMasterPro/issues",
        "Source": "https://github.com/yourusername/LightningMasterPro",
        "Documentation": "https://lightningmasterpro.readthedocs.io/",
    },
)