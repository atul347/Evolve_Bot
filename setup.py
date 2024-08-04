from setuptools import find_packages, setup



setup(
    name="CG_FinBot",
    version="0.0.1",
    author="Atul_Capgemini",
    author_email="atul_deshmukh@live.com",
    packages =find_packages(),
    install_requires=["langchain","langchain-openai","langchain-astradb","datasets","pypdf","python-dotenv","flask"]
)