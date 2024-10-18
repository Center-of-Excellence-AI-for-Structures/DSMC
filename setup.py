from setuptools import setup, find_packages

setup(
    name="monotonic_dc",
    version="1.0",
    description="A robust generalized deep monotonic feature extraction model for label-free prediction of degenerative phenomena.",
    author="Panagiotis Komninos",
    author_email="P.Komninos@tudelft.nl",
    packages=find_packages(include=["dsmc", "dsmc.*"]),
    install_requires=[
        "pandas==1.5.3",
        "matplotlib==3.5.2",
        "seaborn==0.12.1",
        "scikit-learn==1.2.2",
        "scikit-survival==0.21.0",
        "tslearn==0.5.2",
        "joblib==1.2.0",
        "numpy==1.23.4",
        "vallenae==0.7.0",
        "tqdm==4.64.0",
    ],
)
