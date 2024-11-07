from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='SURE',
    version='2.0',
    description='Succinct Representation of Single Cells',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Feng Zeng',
    author_email='zengfeng@xmu.edu.cn',
    packages=find_packages(),
    install_requires=['dill','pytorch-ignite','datatable'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    url='https://github.com/ZengFLab/SUREv2',  # 项目的 GitHub 地址

    entry_points={
        'console_scripts': [
            'SURE=SURE.SURE:main',  # 允许用户通过命令行调用 main 函数
            'SUREMO=SURE.SUREMO:main'
        ],
    },
)