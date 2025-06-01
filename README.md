# Pseudo Factor Analysis (PFA) Implementation

A Python implementation of Pseudo Factor Analysis using transformer embeddings for psychometric research, enabling factor structure exploration before empirical data collection.

## About

This implementation is based on the research paper "Pseudo Factor Analysis of Language Embedding Similarity Matrices: New Ways to Model Latent Constructs" by Guenole et al. It provides tools for analyzing psychological construct structure using language model embeddings, allowing researchers to explore factor structures before collecting any empirical data.

## Key Features

- **Three embedding methods**: Atomic, Atomic Reversed, and Macro approaches
- **Multiple transformer models**: Support for various sentence transformer architectures
- **Comprehensive analysis**: Factor extraction, loading visualization, and method comparison
- **Tucker's congruence**: Statistical comparison between different embedding approaches
- **Robust diagnostics**: Matrix condition analysis and automatic regularization
- **Visualization tools**: Interactive plots for factor loadings and similarity matrices

## Installation

```bash
pip install sentence-transformers factor-analyzer scikit-learn matplotlib seaborn numpy
```

## Quick Start

```python
from pseudo_factor_analysis import PseudoFactorAnalysis

# Initialize with a transformer model
pfa = PseudoFactorAnalysis(model_name='all-MiniLM-L12-v2')

# Define your scale items
items = [
    "I am the life of the party.",
    "I talk to many people at parties.",
    "I feel comfortable around people.",
    "I keep in the background.",
    "I have little to say.",
    "I am always prepared.",
    "I pay attention to details.",
    "I get chores done right away.",
    "I leave my belongings around.",
    "I make a mess of things."
]

# Define item directions (1 = positive, -1 = reversed)
item_signs = [1, 1, 1, -1, -1, 1, 1, 1, -1, -1]

# Run analysis
results = pfa.analyze_scale(
    items=items,
    n_factors=2,
    method='atomic',
    item_signs=item_signs,
    rotation='oblimin'
)

# Visualize results
pfa.plot_loadings(results)
pfa.plot_similarity_matrix(results)
```

## Embedding Methods

### 1. Atomic Method
Embeds each item separately, treating items as independent semantic units.

### 2. Atomic Reversed Method
Embeds items separately and applies directional scoring based on item signs (positive/reversed).

### 3. Macro Method
Embeds each item within the context of the full scale, capturing global semantic relationships.

## Documentation

For comprehensive usage instructions, see the [Implementation Guide](GUIDE.md).

## Example Output

```
--- Analyzing with atomic method ---
Matrix diagnostics: rank=10, condition number=1.13e+01
Factor analysis successful using Principal Axis method
Factor extraction successful
Factors extracted: 2
Eigenvalues: ['2.467', '1.954']
Estimated factor recovery rate: 1.000
Max absolute loadings per factor: ['0.737', '0.896']
```

## Method Comparison

The implementation provides Tucker's congruence coefficients for comparing different embedding approaches:

```python
# Compare different methods
methods = ['atomic', 'atomic_reversed', 'macro']
for method1, method2 in combinations(methods, 2):
    congruence = pfa.tucker_congruence(loadings1, loadings2)
    print(f"{method1} vs {method2}: {np.mean(np.diag(congruence)):.3f}")
```

## Supported Models

- `all-MiniLM-L12-v2` (fast, good general performance)
- `all-mpnet-base-v2` (higher quality, slower)
- `all-distilroberta-base` (BERT-family alternative)
- Any sentence-transformers compatible model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Citation

If you use this implementation in your research, please cite both the original paper and this implementation:

### Original Paper
```bibtex
@article{guenole2024pseudo,
  title={Pseudo Factor Analysis of Language Embedding Similarity Matrices: New Ways to Model Latent Constructs},
  author={Guenole, Nigel and D'Urso, E. Damiano and Samo, Andrew and Sun, Tianjun},
  year={2024},
  note={[This is a pre-print and has not yet been peer reviewed]}
}
```

### This Implementation
```bibtex
@software{shaban2024pfa,
  author = {Shaban, Tariq},
  title = {Pseudo Factor Analysis Implementation},
  year = {2024},
  url = {https://github.com/tariq-shaban/pseudo-factor-analysis},
  note = {Python implementation of PFA methodology}
}
```

## Author

**Tariq Shaban**  
Email: tariq@inubilum.io  
Organization: Inubilum

## Original Research

This implementation is based on the methodology described in:

> Guenole, N., D'Urso, E. D., Samo, A., & Sun, T. (2024). Pseudo Factor Analysis of Language Embedding Similarity Matrices: New Ways to Model Latent Constructs. *[Pre-print]*

**Original Authors:** Nigel Guenole (Goldsmiths, University of London), E. Damiano D'Urso (Independent Researcher), Andrew Samo (Bowling Green State University), Tianjun Sun (Rice University)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Tariq Shaban

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Disclaimer

This software is provided for research and educational purposes. While based on peer-reviewed methodology, this implementation is provided "as is" without warranty. Users should validate results against empirical data when possible.

---

⭐ If you find this implementation useful, please consider starring the repository and citing our work!
