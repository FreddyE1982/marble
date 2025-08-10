# Dataset Encryption

MARBLE secures datasets using **AES-256-GCM** to protect data at rest and in
transit. Keys are stored outside the repository and supplied via the
`DATASET_ENCRYPTION_KEY` environment variable.

## Key Management

- Keys are 256-bit and base64 encoded for transport.
- Generate a new key with:

  ```python
  from dataset_encryption import generate_key
  print(generate_key())
  ```
- Store the value in a secure secret manager and expose it at runtime
  through the `DATASET_ENCRYPTION_KEY` variable.

## Encrypting Tensors

```python
import torch
from dataset_encryption import load_key_from_env, encrypt_tensor, decrypt_tensor

key = load_key_from_env()
original = torch.rand(2, 3, device="cuda" if torch.cuda.is_available() else "cpu")
enc = encrypt_tensor(original, key)
restored = decrypt_tensor(enc, key)
assert torch.allclose(original, restored)
```

The helper functions automatically move tensors to CPU for encryption and
restore them to the requested device during decryption.
