from setuptools import find_packages, setup

print(find_packages(exclude=('configs', 'experiments', 'tools')))

if __name__ == '__main__':
    setup(
        name='fovea',
        packages=find_packages(exclude=('configs', 'experiments', 'tools')),
    )
