# File location: setup.py

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent

long_description = (ROOT / "README.md").read_text(encoding="utf-8")

requirements = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]

EXTRAS = {
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=24.1.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.8.0",
        "pre-commit>=3.5.0",
    ],
    "docs": [
        "sphinx>=7.1.0",
        "sphinx-rtd-theme>=1.3.0",
    ],
    "export": [
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        "onnxscript>=0.1.0",
    ],
    "notebooks": [
        "jupyter>=1.0.0",
        "ipykernel>=6.25.0",
    ],
}
EXTRAS["all"] = sorted({dep for deps in EXTRAS.values() for dep in deps})

setup(
    name="lightning-master-pro",
    version="0.2.0",
    author="Satvik Praveen",
    author_email="satvikpraveen707@gmail.com",
    description="A hands-on PyTorch Lightning learning framework: modules, datamodules, callbacks, loops and CLI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SatvikPraveen/LightningMasterPro",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require=EXTRAS,
    entry_points={
        "console_scripts": [
            "lmpro=lmpro.cli:main",
        ],
    },
    package_data={"lmpro": ["py.typed"]},
    include_package_data=True,
    project_urls={
        "Bug Reports": "https://github.com/SatvikPraveen/LightningMasterPro/issues",
        "Source": "https://github.com/SatvikPraveen/LightningMasterPro",
    },
)
