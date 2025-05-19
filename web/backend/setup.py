from setuptools import setup, find_packages

setup(
    name="backend",
    # 其他配置将从pyproject.toml读取
    packages=find_packages(exclude=["tests*", "tmp*"])
)