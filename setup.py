from setuptools import setup, find_packages

setup(
    name="pyvroomm",                    # Replace with your module name
    version="1.0.0",
    author="Jonathan St-Antoine",                 # Replace with your name
    author_email="jonathan@astro.umontreal.ca",      # Replace with your email
    description="VROOMM data modeling",
    url="",  # Optional
    
    packages=find_packages(),           # Automatically find all packages
    python_requires=">=3.12",            # Minimum Python version
    
    # Add dependencies here if needed
    install_requires=[
        "natsort",
        # "numpy",
    ],
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
