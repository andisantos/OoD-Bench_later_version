from setuptools import setup, find_packages

_VERSION = '1.0'

setup(
    name='domainbed',
    version=_VERSION,
    packages=find_packages(),
    py_modules=['algorithms',
                'command_launchers'
                'datasets',
                'hparams_registry',
                'model_selection',
                'networks']
)
