import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from distutils import log
from typing import List, Dict

# Constants
PROJECT_NAME = "enhanced_stat.ML_2508.15678v1_Tree_like_Pairwise_Interaction_Networks"
VERSION = "1.0.0"
AUTHOR = "Your Name"
AUTHOR_EMAIL = "your@email.com"
DESCRIPTION = "Enhanced AI project based on stat.ML_2508.15678v1_Tree-like-Pairwise-Interaction-Networks with content analysis"
LONG_DESCRIPTION = "Detected project type: agent (confidence score: 6 matches)"
KEYWORDS = ["agent", "tree-like", "pairwise", "interaction", "networks"]
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]
INSTALL_REQUIRES = [
    "torch",
    "numpy",
    "pandas",
]
EXTRAS_REQUIRE = {
    "dev": ["pytest", "flake8", "mypy"],
}
ENTRY_POINTS = {
    "console_scripts": [
        "agent=agent.main:main",
    ],
}

# Setup function
def setup_package():
    setup(
        name=PROJECT_NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        entry_points=ENTRY_POINTS,
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False,
    )

# Custom install command
class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        log.info("Running custom install command")

# Custom develop command
class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)
        log.info("Running custom develop command")

# Custom egg info command
class CustomEggInfoCommand(egg_info):
    def run(self):
        egg_info.run(self)
        log.info("Running custom egg info command")

# Setup commands
cmdclass = {
    "install": CustomInstallCommand,
    "develop": CustomDevelopCommand,
    "egg_info": CustomEggInfoCommand,
}

# Main setup function
def main():
    setup_package()

if __name__ == "__main__":
    main()