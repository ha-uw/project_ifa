# IfáDTA

A comprehensive toolkit for handling various machine learning methods and datasets related to drug-target interaction (DTI).

## Table of Contents 📚

- [Overview](#overview-📊)
- [What is Ifá?](#what-is-ifá-🌟)
- [Installation](#installation-🛠️)
- [Usage](#usage-🚀)
- [Features](#features-✨)
- [Contributing](#contributing-🤝)
- [License](#license-📄)

## Overview 📊

This project is designed to handle various data processing and analysis tasks related to drug-target interaction (DTI) datasets. It includes multiple configurations and methods for different DTI prediction models such as DeepDTA, GraphDTA, and WideDTA.

### What is Ifá? 🌟

Ifá (ee-fah) is a system of divination that originated in West Africa, particularly among the Yoruba people. It involves the use of wisdom and knowledge to provide guidance and insights. In the context of this project, the name "IfáDTA" symbolizes the project's goal of using advanced data processing and machine learning techniques to gain insights and make predictions about drug-target interactions, much like how Ifá provides guidance and wisdom.

## Installation 🛠️

To set up the environment, use the provided `environment.yml` file. You can create a new conda environment with the following command:

```sh
conda env create -f environment.yml
```

Activate the environment:

```sh
conda activate project_ifa
```

## Usage 🚀

The main entry point for the project is the ifa.py script. To run the project, execute the following command:

```sh
python ifa.py
```

This will invoke the [`main`](command:_github.copilot.openSymbolFromReferences?%5B%22main%22%2C%5B%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22c%3A%5C%5CUsers%5C%5CRDV01%5C%5Ccode%5C%5Cbbk%5C%5Cproject_ifa%5C%5Cifa.py%22%2C%22_sep%22%3A1%2C%22external%22%3A%22file%3A%2F%2F%2Fc%253A%2FUsers%2FRDV01%2Fcode%2Fbbk%2Fproject_ifa%2Fifa.py%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2FRDV01%2Fcode%2Fbbk%2Fproject_ifa%2Fifa.py%22%2C%22scheme%22%3A%22file%22%7D%2C%22pos%22%3A%7B%22line%22%3A5%2C%22character%22%3A20%7D%7D%5D%5D "Go to definition") function from the [`src/cli.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2FRDV01%2Fcode%2Fbbk%2Fproject_ifa%2Fsrc%2Fcli.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "c:\\Users\RDV01\code\bbk\project_ifa\src\cli.py") file, which handles command-line arguments and orchestrates the execution of the program.

## Project Structure 🗂️

- **configs/**: Configuration files for different DTI models.
  - `DeepDTA/`
  - `GraphDTA/`
  - `WideDTA/`
- **data/**: Contains various datasets.
  - `bindingdb_ic50/`
  - `bindingdb_kd/`
  - `bindingdb_ki/`
  - `davis/`
  - `kiba/`
- **outputs/**: Output files and results from different models.
  - `DeepDTA/`
  - `GraphDTA/`
  - `GraphDTA_GAT/`
  - `GraphDTA_GIN/`
  - `WideDTA/`
- **src/**: Source code for the project.
  - `cli.py`: Command-line interface for the project.
  - `data/`: Data processing scripts.
- **tests/**: Unit tests for the project.
  - `test_encoders.py`
  - `test_loading.py`
  - `test_processing.py`
  - `test_trainers.py`

## Running Tests 🧪

To run the tests, use the following command:

```sh
python -m unittest tests/
```

## Contributing 🤝

If you wish to contribute to this project, please fork the repository and submit a pull request. Ensure that all tests pass and adhere to the project's coding standards.

## License 📄

```
This project is licensed under the MIT License. See the LICENSE file for more details.
```
