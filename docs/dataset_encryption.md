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

## Encrypting Cached Files

Dataset utilities such as :mod:`dataset_loader` can encrypt files on disk
using the same AES-256-GCM scheme. Pass the base64-encoded key to
``prefetch_dataset``, ``load_dataset`` or ``export_dataset`` and the resulting
bytes will be written as encrypted blobs::

```python
from dataset_encryption import load_key_from_env
from dataset_loader import prefetch_dataset, wait_for_prefetch, export_dataset

key = load_key_from_env()
prefetch_dataset("https://example.com/data.csv", encryption_key=key)
wait_for_prefetch()  # cached file is now encrypted at rest
export_dataset([(1,2)], "data.csv", encryption_key=key)
```

Files are transparently decrypted when loading the dataset as long as the same
key is supplied.
