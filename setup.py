from setuptools import setup, find_packages

setup(
    name='sampi',
    version='0.0.1',
    author='Jonathan Lindbloom',
    author_email='jonathan@lindbloom.com',
    license='LICENSE',
    packages=find_packages(),
    description='Some Monte Carlo codes.',
    long_description=open('README.md').read(),
)