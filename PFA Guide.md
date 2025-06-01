# Pseudo Factor Analysis (PFA) Implementation Guide

This guide explains how to use the Pseudo Factor Analysis implementation based on the paper "Pseudo Factor Analysis of Language Embedding Similarity Matrices: New Ways to Model Latent Constructs" by Guenole et al.

## Overview

Pseudo Factor Analysis (PFA) is a novel approach that uses language model embeddings to analyze the latent structure of psychological constructs **before collecting any empirical data**. Instead of relying on response correlations, PFA analyzes the semantic similarity patterns between scale items using transformer-based language models.

## Key Concepts

### What is Pseudo Factor Analysis?

PFA replaces the traditional item-response correlation matrix with a **cosine similarity matrix** computed from language embeddings of scale items. This allows researchers to:

- Explore factor structures before data collection
- Validate scale construction using semantic relationships
- Compare different embedding approaches for psychometric analysis
- Bridge conventional psychometrics with modern NLP techniques

### Three Embedding Methods

The implementation supports three encoding approaches from the original paper:

## 1. Atomic Method
```python
# Each item embedded separately
items = ["I am outgoing", "I am shy", "I am organized"]
embeddings = model.encode(items)  # Each item gets its own embedding
```

**Use case**: Standard approach where each item is treated independently.

## 2. Atomic Reversed Method
```python
# Items embedded separately, then multiplied by directional signs
item_signs = [1, -1, 1]  # positive, reversed, positive
embeddings = model.encode(items) * item_signs.reshape(-1, 1)
```

**Use case**: When you have reversed-scored items and want to account for their directional relationship to the construct.

## 3. Macro Method
```python
# Each item embedded with full scale context
for item in items:
    contextualized = f"Scale context: {all_items} || Focus item: {item}"
    embedding = model.encode([contextualized])
```

**Use case**: Captures global scale-level information while preserving individual item differences.

## Getting Started

### Installation Requirements

```python
pip install sentence-transformers factor-analyzer scikit-learn matplotlib seaborn numpy
```

### Basic Usage

```python
from pseudo_factor_analysis import PseudoFactorAnalysis

# Initialize with a transformer model
pfa = PseudoFactorAnalysis(model_name='all-MiniLM-L12-v2')

# Define your scale items
items = [
    "I am the life of the party.",           # Extraversion (+)
    "I talk to many people at parties.",     # Extraversion (+) 
    "I feel comfortable around people.",     # Extraversion (+)
    "I keep in the background.",             # Extraversion (-)
    "I have little to say.",                 # Extraversion (-)
    "I am always prepared.",                 # Conscientiousness (+)
    "I pay attention to details.",           # Conscientiousness (+)
    "I get chores done right away.",         # Conscientiousness (+)
    "I leave my belongings around.",         # Conscientiousness (-)
    "I make a mess of things."               # Conscientiousness (-)
]

# Define item directions (1 = positive, -1 = reversed)
item_signs = [1, 1, 1, -1, -1, 1, 1, 1, -1, -1]

# Run analysis
results = pfa.analyze_scale(
    items=items,
    n_factors=2,
    method='atomic',  # or 'atomic_reversed' or 'macro'
    item_signs=item_signs,
    rotation='oblimin'
)
```

## Understanding the Output

### Factor Analysis Results

```python
# Access factor loadings
loadings = results['factor_analysis']['loadings']
print(f"Factor loadings shape: {loadings.shape}")  # (n_items, n_factors)

# Eigenvalues indicate factor strength
eigenvals = results['factor_analysis']['eigenvals']
print(f"Eigenvalues: {eigenvals}")

# Variance explained by each factor
variance = results['factor_analysis']['variance_explained']
print(f"Variance explained: {variance}")
```

### Interpreting Factor Loadings

```python
# Items with |loading| > 0.3 are considered to load on a factor
threshold = 0.3
for factor in range(loadings.shape[1]):
    factor_loadings = loadings[:, factor]
    high_loading_items = np.where(np.abs(factor_loadings) >= threshold)[0]
    
    print(f"Factor {factor + 1}:")
    for item_idx in high_loading_items:
        loading = factor_loadings[item_idx]
        print(f"  Item {item_idx}: {items[item_idx][:50]}... (loading: {loading:.3f})")
```

## Method Comparison

### Comparing Different Embedding Approaches

```python
methods = ['atomic', 'atomic_reversed', 'macro']
all_results = {}

for method in methods:
    results = pfa.analyze_scale(
        items=items,
        n_factors=2,
        method=method,
        item_signs=item_signs if method == 'atomic_reversed' else None
    )
    all_results[method] = results

# Calculate Tucker's congruence between methods
from itertools import combinations

for method1, method2 in combinations(methods, 2):
    loadings1 = all_results[method1]['factor_analysis']['loadings']
    loadings2 = all_results[method2]['factor_analysis']['loadings']
    
    congruence = pfa.tucker_congruence(loadings1, loadings2)
    mean_congruence = np.mean(np.diag(congruence))
    
    print(f"{method1} vs {method2}: {mean_congruence:.3f}")
```

### Congruence Interpretation

- **> 0.95**: Excellent similarity
- **0.85-0.95**: Fair similarity  
- **< 0.85**: Poor similarity

## Visualization

### Plot Factor Loadings

```python
pfa.plot_loadings(results, figsize=(10, 8))
```

Creates a heatmap showing how strongly each item loads on each factor.

### Plot Similarity Matrix

```python
pfa.plot_similarity_matrix(results, figsize=(10, 8))
```

Shows the cosine similarity between all pairs of items.

## Advanced Usage

### Model Selection

Different transformer models can yield different results:

```python
models_to_test = [
    'all-MiniLM-L12-v2',        # Fast, good general performance
    'all-mpnet-base-v2',        # Better quality, slower
    'all-distilroberta-base'    # BERT-family alternative
]

for model_name in models_to_test:
    pfa = PseudoFactorAnalysis(model_name=model_name)
    results = pfa.analyze_scale(items, n_factors=2)
    # Compare results...
```

### Matrix Diagnostics

The implementation provides diagnostic information:

```python
# Check matrix properties
eigenvals = np.linalg.eigvals(similarity_matrix)
rank = np.linalg.matrix_rank(similarity_matrix)
condition_number = np.linalg.cond(similarity_matrix)

print(f"Matrix rank: {rank}")
print(f"Condition number: {condition_number:.2e}")
print(f"Minimum eigenvalue: {np.min(eigenvals):.2e}")
```

### Handling Problematic Matrices

The code automatically handles common issues:

- **Ill-conditioned matrices**: Adds regularization
- **Non-positive definite matrices**: Applies stabilization
- **Convergence failures**: Falls back to alternative factor extraction methods

## Best Practices

### 1. Item Selection
- Use clear, unambiguous item wording
- Include both positively and negatively worded items
- Ensure items represent the intended constructs

### 2. Method Choice
- **Atomic**: Start here for most applications
- **Atomic_reversed**: Use when you have reversed items and clear directional expectations
- **Macro**: Try when you want to capture scale-level context

### 3. Factor Interpretation
- Look for items with |loadings| > 0.3
- Check that factors make theoretical sense
- Compare results across different methods
- Validate with empirical data when possible

### 4. Model Selection
- Start with `all-MiniLM-L12-v2` for speed
- Try `all-mpnet-base-v2` for higher quality
- Consider domain-specific models for specialized content

## Limitations and Considerations

### From the Paper
- **No sample size requirements**: Unlike traditional FA, PFA doesn't need empirical data
- **Model dependency**: Results depend on the quality and training of the language model
- **Semantic vs. Empirical**: Captures semantic relationships, which may differ from behavioral relationships
- **Validation needed**: Should be validated against empirical factor structures when possible

### Practical Limitations
- **Computational cost**: Larger models require more resources
- **Language dependency**: Models trained on English may not work well for other languages
- **Context sensitivity**: Results can vary based on item wording and context

## Example: Complete Analysis Pipeline

```python
def complete_pfa_analysis(items, item_signs, n_factors=2):
    """Complete PFA analysis with all methods and comparisons."""
    
    pfa = PseudoFactorAnalysis('all-MiniLM-L12-v2')
    methods = ['atomic', 'atomic_reversed', 'macro']
    results = {}
    
    # Run all methods
    for method in methods:
        print(f"\n--- {method.upper()} METHOD ---")
        
        result = pfa.analyze_scale(
            items=items,
            n_factors=n_factors,
            method=method,
            item_signs=item_signs if method == 'atomic_reversed' else None
        )
        
        results[method] = result
        
        # Print key results
        loadings = result['factor_analysis']['loadings']
        eigenvals = result['factor_analysis']['eigenvals']
        
        print(f"Eigenvalues: {eigenvals[:n_factors]}")
        print(f"Variance explained: {result['factor_analysis']['variance_explained']}")
    
    # Method comparisons
    print(f"\n--- METHOD COMPARISONS ---")
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            congruence = pfa.tucker_congruence(
                results[method1]['factor_analysis']['loadings'],
                results[method2]['factor_analysis']['loadings']
            )
            print(f"{method1} vs {method2}: {np.mean(np.diag(congruence)):.3f}")
    
    return results, pfa

# Run complete analysis
results, pfa_instance = complete_pfa_analysis(items, item_signs)

# Generate visualizations
for method in ['atomic', 'atomic_reversed', 'macro']:
    print(f"\nPlots for {method} method:")
    pfa_instance.plot_loadings(results[method])
    pfa_instance.plot_similarity_matrix(results[method])
```

This implementation provides a comprehensive tool for exploring psychological construct structure using modern NLP techniques, bridging traditional psychometrics with contemporary language understanding capabilities.
