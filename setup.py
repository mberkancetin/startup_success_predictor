from setuptools import setup, find_packages


with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(
    name="predictor",
    description="startup success predictor package",
    packages=find_packages(),
    install_requires=requirements
)

print("setup completed")
