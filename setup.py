from setuptools import setup, find_packages

setup(
    name="hosting-chatbot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain==0.0.350",
        "langchain-community==0.0.13",
        "langchain-openai==0.0.5",
        "openai==1.12.0",
        "faiss-cpu==1.8.0",
        "python-dotenv==1.0.1",
        "pandas==2.2.1",
        "numpy==1.26.4",
        "tqdm==4.66.2",
        "pydantic==2.6.3",
        "pydantic-settings==2.2.1",
        "python-dateutil==2.8.2",
        "urllib3<2.0.0",
        "httpx==0.27.2"
    ],
    python_requires=">=3.11",
) 