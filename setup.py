from pkg_resources import parse_requirements
from setuptools import find_packages, setup


def main():
    package_name = 'samogonka'
    packages = find_packages(package_name)
    packages = list(map(lambda x: f'{package_name}/{x}', packages))
    reqs = []#[str(req) for req in parse_requirements(open('requirements.txt'))]

    setup(
        name=package_name,
        version='0.0.1',
        author='sergevkim',
        description=package_name,
        package_dir={package_name: package_name},
        packages=packages,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.9',
        install_requires=reqs,
    )


if __name__ == '__main__':
    main()