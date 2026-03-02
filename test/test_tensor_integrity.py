import torch

# Tensor file paths
TENSOR_PATH = ['temp/roberta_bio_embeddings.pt', 'temp/roberta_embeddings.pt']

for path in TENSOR_PATH:
    print(f"\nchecking file: {path}")
    data = torch.load(path, map_location='cpu', weights_only=False)

    X = data['embeddings'].float()
    y = data['labels'].long()

    print("\n=== 1. Shapes ===")
    print(f"Shape of matrix X (embeddings): {X.shape} -> Expected [N, 768]")
    print(f"Shape of matrix Y (labels):     {y.shape} -> Expected [N]")

    # NaN (Not a Number) or Inf (Infinity) values can break training. // Expected False for both
    print("\n=== 2. Corruption (NaN / Inf) ===")
    has_nan = torch.isnan(X).any().item()
    has_inf = torch.isinf(X).any().item()
    print(f"Includes NaN values?: {has_nan} -> Expected False")
    print(f"Includes Inf values?: {has_inf} -> Expected False")

    # If RoBERTa fails to generate embeddings, we might get many zero vectors.
    # few is normal (users without bios)
    print("\n=== 3. Zero vectors ===")
    zero_vectors = (X == 0).all(dim=1).sum().item()
    print(f"Number of zero vectors: {zero_vectors} ({(zero_vectors/X.shape[0])*100:.2f} %)")

    # Are the vectors diverse? 
    # If RoBERTa fails, all vectors might be identical (or very similar)
    print("\n=== 4. Standard deviation test ===")
    std_dev = torch.std(X, dim=0).mean().item()
    print(f"Standard deviation: {std_dev:.4f} -> Expected > 0.0")

    # Check if the first two vectors are identical 
    are_identical = torch.allclose(X[0], X[1])
    print(f"Are vectors of first two users identical?: {are_identical} -> Expected False")


    print("\n=== 5. Example ===")
    print("First 5:")
    print(X[0, :5].tolist())


    print("\n" + "="*60 + "\n")
    print("\n" + "="*60 + "\n")





##############################################

