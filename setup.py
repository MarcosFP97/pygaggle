import setuptools


with open('README.md') as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name='pygaggle',
    version='0.0.1',
    author='PyGaggle Gaggle',
    author_email='r33tang@uwaterloo.ca',
    description='A gaggle of rerankers for CovidQA, CORD-19 and MS-MARCO',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/castorini/pygaggle',
    install_requires=requirements,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
