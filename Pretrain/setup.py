from setuptools import setup, find_packages

packages = find_packages(where=".")

setup(
    packages=packages,
    entry_points={
        'console_scripts': [
            'amamba2_train=audiotrain.methods.clip.train:main',
            'amamba2_freeze=audiotrain.methods.clip.downstream.train_freeze:main',
            'amamba2_finetune=audiotrain.methods.clip.downstream.train_finetune:main'
        ]
    }
)
