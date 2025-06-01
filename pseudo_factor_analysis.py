"""
Pseudo Factor Analysis Implementation

A Python implementation of Pseudo Factor Analysis using transformer embeddings
for psychometric research, based on the methodology described in:

    Guenole, N., D'Urso, E. D., Samo, A., & Sun, T. (2024). 
    Pseudo Factor Analysis of Language Embedding Similarity Matrices: 
    New Ways to Model Latent Constructs.

Implementation Author: Tariq Shaban
Contact: tariq@inubilum.io
Organization: Inubilum

Copyright (c) 2025 Tariq Shaban
Licensed under the MIT License - see LICENSE file for details.

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
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from factor_analyzer import FactorAnalyzer
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PseudoFactorAnalysis:
    """
    Pseudo Factor Analysis implementation using language model embeddings
    to analyze latent constructs in psychological scales.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L12-v2'):
        """
        Initialize PFA with a sentence transformer model.
        
        Args:
            model_name: HuggingFace sentence transformer model name
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embeddings = None
        self.similarity_matrix = None
        self.factor_loadings = None
        self.items = None
        self.item_signs = None
        
    def encode_items(self, items: List[str], method: str = 'atomic', 
                    item_signs: Optional[List[int]] = None) -> np.ndarray:
        """
        Encode scale items using different approaches.
        
        Args:
            items: List of scale items (strings)
            method: 'atomic', 'atomic_reversed', or 'macro'
            item_signs: List of 1 or -1 for each item (for reversed scoring)
            
        Returns:
            Embeddings matrix
        """
        self.items = items
        self.item_signs = item_signs if item_signs else [1] * len(items)
        
        if method == 'atomic':
            # Embed each item separately
            embeddings = self.model.encode(items)
            
        elif method == 'atomic_reversed':
            # Embed each item and multiply by its sign
            embeddings = self.model.encode(items)
            if item_signs:
                embeddings = embeddings * np.array(item_signs).reshape(-1, 1)
            
        elif method == 'macro':
            # Proper macro approach: concatenate all items and embed together
            # Then use this as context for individual item embeddings
            concatenated_context = ' '.join(items)
            
            # Method 1: Embed each item with its position in the concatenated context
            embeddings = []
            for i, item in enumerate(items):
                # Create a contextualized version where this item is highlighted
                item_with_context = f"Scale context: {concatenated_context} || Focus item: {item}"
                embedding = self.model.encode([item_with_context])
                embeddings.append(embedding[0])
            embeddings = np.array(embeddings)
            
        else:
            raise ValueError("Method must be 'atomic', 'atomic_reversed', or 'macro'")
            
        self.embeddings = embeddings
        return embeddings
    
    def compute_similarity_matrix(self, embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute cosine similarity matrix from embeddings.
        
        Args:
            embeddings: Item embeddings (uses self.embeddings if None)
            
        Returns:
            Cosine similarity matrix
        """
        if embeddings is None:
            embeddings = self.embeddings
            
        if embeddings is None:
            raise ValueError("No embeddings found. Run encode_items first.")
            
        similarity_matrix = cosine_similarity(embeddings)
        self.similarity_matrix = similarity_matrix
        return similarity_matrix
    
    def fit_factor_analysis(self, n_factors: int, rotation: str = 'oblimin', 
                           similarity_matrix: Optional[np.ndarray] = None) -> Dict:
        """
        Perform factor analysis on the similarity matrix.
        
        Args:
            n_factors: Number of factors to extract
            rotation: Rotation method ('oblimin' for oblique as in paper, 'varimax', 'promax', etc.)
            similarity_matrix: Similarity matrix (uses self.similarity_matrix if None)
            
        Returns:
            Dictionary with factor analysis results
        """
        if similarity_matrix is None:
            similarity_matrix = self.similarity_matrix
            
        if similarity_matrix is None:
            raise ValueError("No similarity matrix found. Run compute_similarity_matrix first.")
        
        # Add small values to diagonal to improve numerical stability
        stabilized_matrix = similarity_matrix.copy()
        np.fill_diagonal(stabilized_matrix, 1.0)
        
        # Check matrix properties and add diagnostics
        eigenvals = np.linalg.eigvals(stabilized_matrix)
        rank = np.linalg.matrix_rank(stabilized_matrix)
        condition_number = np.linalg.cond(stabilized_matrix)
        
        print(f"Matrix diagnostics: rank={rank}, condition number={condition_number:.2e}")
        
        # Check for problematic matrices
        if rank < min(stabilized_matrix.shape[0], n_factors):
            print(f"Warning: Matrix rank ({rank}) is less than number of factors ({n_factors})")
            
        if condition_number > 1e12:
            print(f"Warning: Matrix is ill-conditioned (condition number: {condition_number:.2e})")
            
        if np.min(eigenvals) <= 1e-10:
            print(f"Warning: Matrix not positive definite (min eigenvalue: {np.min(eigenvals):.2e}). Adding regularization.")
            regularization = max(1e-6, -np.min(eigenvals) + 1e-6)
            stabilized_matrix += np.eye(stabilized_matrix.shape[0]) * regularization
            print(f"Added regularization: {regularization:.2e}")
        
        # Try maximum likelihood first (as in paper), fallback to principal if it fails
        methods_to_try = [
            ('ml', 'Maximum Likelihood'),
            ('principal', 'Principal Axis'),
            ('minres', 'Minimum Residual')
        ]
        
        for method, method_name in methods_to_try:
            try:
                fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method)
                fa.fit(stabilized_matrix)
                
                print(f"Factor analysis successful using {method_name} method")
                
                results = {
                    'loadings': fa.loadings_,
                    'communalities': fa.get_communalities(),
                    'uniquenesses': fa.get_uniquenesses(),
                    'eigenvals': fa.get_eigenvalues()[0],
                    'variance_explained': fa.get_factor_variance(),
                    'rotation': rotation,
                    'n_factors': n_factors,
                    'method_used': method,
                    'factor_correlations': getattr(fa, 'phi_', None) if rotation in ['oblimin', 'promax'] else None
                }
                
                self.factor_loadings = fa.loadings_
                return results
                
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Warning: {method_name} method failed ({str(e)}), trying next method...")
                continue
        
        # If all methods fail, raise an error
        raise RuntimeError("All factor analysis methods failed. Check your similarity matrix.")
    
    def analyze_scale(self, items: List[str], n_factors: int, 
                     method: str = 'atomic', item_signs: Optional[List[int]] = None,
                     rotation: str = 'oblimin') -> Dict:
        """
        Complete PFA analysis pipeline.
        
        Args:
            items: List of scale items
            n_factors: Number of factors to extract
            method: Encoding method
            item_signs: Item direction signs
            rotation: Factor rotation method
            
        Returns:
            Complete analysis results
        """
        # Step 1: Encode items
        embeddings = self.encode_items(items, method, item_signs)
        
        # Step 2: Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        
        # Step 3: Factor analysis
        fa_results = self.fit_factor_analysis(n_factors, rotation, similarity_matrix)
        
        # Step 4: Compile results
        results = {
            'method': method,
            'model_name': self.model_name,
            'items': items,
            'embeddings': embeddings,
            'similarity_matrix': similarity_matrix,
            'factor_analysis': fa_results,
            'item_signs': self.item_signs
        }
        
        return results
    
    def plot_loadings(self, results: Dict, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot factor loadings heatmap.
        
        Args:
            results: Results from analyze_scale()
            figsize: Figure size
        """
        loadings = results['factor_analysis']['loadings']
        items = results['items']
        
        # Create labels for factors
        n_factors = loadings.shape[1]
        factor_labels = [f'Factor {i+1}' for i in range(n_factors)]
        
        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(loadings, 
                   xticklabels=factor_labels,
                   yticklabels=[f'Item {i+1}' for i in range(len(items))],
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   fmt='.3f')
        plt.title(f'Factor Loadings - {results["method"].title()} Method')
        plt.xlabel('Factors')
        plt.ylabel('Items')
        plt.tight_layout()
        plt.show()
    
    def plot_similarity_matrix(self, results: Dict, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot item similarity matrix.
        
        Args:
            results: Results from analyze_scale()
            figsize: Figure size
        """
        similarity_matrix = results['similarity_matrix']
        items = results['items']
        
        plt.figure(figsize=figsize)
        sns.heatmap(similarity_matrix,
                   xticklabels=[f'Item {i+1}' for i in range(len(items))],
                   yticklabels=[f'Item {i+1}' for i in range(len(items))],
                   annot=True,
                   cmap='viridis',
                   fmt='.3f')
        plt.title(f'Item Similarity Matrix - {results["method"].title()} Method')
        plt.tight_layout()
        plt.show()
    
    def tucker_congruence(self, loadings1: np.ndarray, loadings2: np.ndarray) -> np.ndarray:
        """
        Calculate Tucker's congruence coefficient between two loading matrices.
        
        Args:
            loadings1: First loading matrix
            loadings2: Second loading matrix
            
        Returns:
            Matrix of congruence coefficients
        """
        n_factors1 = loadings1.shape[1]
        n_factors2 = loadings2.shape[1]
        congruence_matrix = np.zeros((n_factors1, n_factors2))
        
        for i in range(n_factors1):
            for j in range(n_factors2):
                factor1 = loadings1[:, i]
                factor2 = loadings2[:, j]
                
                numerator = np.sum(factor1 * factor2)
                denominator = np.sqrt(np.sum(factor1**2) * np.sum(factor2**2))
                
                if denominator != 0:
                    congruence_matrix[i, j] = numerator / denominator
                else:
                    congruence_matrix[i, j] = 0
                    
        return congruence_matrix

# Example usage and demonstration
def demo_pfa():
    """Demonstration of PFA with sample personality items, following the paper's approach."""
    
    # Sample personality items with diverse content and some reversed items
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
    
    # Item signs (1 for positive, -1 for reversed items)
    item_signs = [1, 1, 1, -1, -1, 1, 1, 1, -1, -1]
    
    # Test different models as in the paper
    models_to_test = [
        'all-MiniLM-L12-v2',  # Similar to all-MiniLM-L12-v2 in paper
        'all-distilroberta-base',  # Similar to distilroberta in paper
        'sentence-transformers/all-mpnet-base-v2'  # Similar to MPNet in paper
    ]
    
    print("Running Pseudo Factor Analysis Demo...")
    print("=" * 60)
    
    # Test different methods as in the paper
    methods = ['atomic', 'atomic_reversed', 'macro']
    all_results = {}
    
    for model_name in models_to_test[:1]:  # Test one model for demo
        print(f"\nUsing model: {model_name}")
        pfa = PseudoFactorAnalysis(model_name=model_name)
        
        model_results = {}
        
        for method in methods:
            print(f"\n--- Analyzing with {method} method ---")
            
            results = pfa.analyze_scale(
                items=items,
                n_factors=2,  # Extracting 2 factors (Extraversion, Conscientiousness)
                method=method,
                item_signs=item_signs if method == 'atomic_reversed' else None,
                rotation='oblimin'  # Oblique rotation as in paper
            )
            
            model_results[method] = results
            
            # Print results in paper style
            loadings = results['factor_analysis']['loadings']
            eigenvals = results['factor_analysis']['eigenvals']
            n_factors_extracted = loadings.shape[1]
            
            print(f"Factor extraction successful")
            print(f"Factors extracted: {n_factors_extracted}")
            print(f"Eigenvalues: {[f'{e:.3f}' for e in eigenvals[:n_factors_extracted]]}")
            
            # Calculate global factor structure recovery (simplified)
            # Check if we have meaningful factor structure
            max_loadings = np.max(np.abs(loadings), axis=0)
            n_recovered = np.sum(max_loadings > 0.3)
            recovery_rate = n_recovered / n_factors_extracted
            print(f"Estimated factor recovery rate: {recovery_rate:.3f}")
            print(f"Max absolute loadings per factor: {[f'{ml:.3f}' for ml in max_loadings]}")
            
            # Check for potential issues
            if np.all(max_loadings < 0.3):
                print("Warning: No strong factor loadings found (all |loadings| < 0.3)")
            if len(set([f'{ml:.2f}' for ml in max_loadings])) == 1:
                print("Warning: All factors have identical maximum loadings - possible numerical issue")
            
            # Show dominant loadings per factor (paper style)
            print("Factor loadings (|loading| > 0.3):")
            for factor in range(n_factors_extracted):
                factor_loadings = loadings[:, factor]
                high_items = np.where(np.abs(factor_loadings) > 0.3)[0]
                print(f"  Factor {factor+1}: Items {high_items} "
                      f"(loadings: {[f'{factor_loadings[i]:.3f}' for i in high_items]})")
        
        # Compare methods using Tucker's congruence (as in paper)
        print(f"\n--- Method Comparison (Tucker's Congruence) ---")
        method_names = list(model_results.keys())
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i+1:]:
                loadings1 = model_results[method1]['factor_analysis']['loadings']
                loadings2 = model_results[method2]['factor_analysis']['loadings']
                
                congruence = pfa.tucker_congruence(loadings1, loadings2)
                mean_congruence = np.mean(np.diag(congruence))
                
                print(f"{method1} vs {method2}: Mean congruence = {mean_congruence:.3f}")
        
        all_results[model_name] = model_results
    
    return all_results, pfa

if __name__ == "__main__":
    # Run demonstration
    results, pfa_instance = demo_pfa()
    
    # Plot results for atomic method from the first model
    model_name = list(results.keys())[0]
    model_results = results[model_name]
    
    if 'atomic' in model_results:
        print("\nGenerating plots for atomic method...")
        pfa_instance.plot_similarity_matrix(model_results['atomic'])
        pfa_instance.plot_loadings(model_results['atomic'])
