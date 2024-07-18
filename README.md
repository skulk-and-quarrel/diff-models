# LLM Logit Comparison
This project implements a token generation method based on logit differences between two language models.

## Installation
```
pip install -r requirements.txt
```

## Usage

Run the basic comparison example:
```
python main.py
```

## Project Structure

- `src/`: Contains the main package code
  - `models/`: Model loading utilities
  - `processors/`: Custom logits processors
  - `utils/`: Utility functions for token generation
- `examples/`: Example scripts
- `tests/`: Unit tests (to be implemented)