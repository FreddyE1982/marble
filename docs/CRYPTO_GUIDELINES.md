# Cryptographic Safety Guidelines

MARBLE uses cryptographic primitives when communicating between nodes. To prevent
timing attacks all secret comparisons must use constant-time algorithms. The
`crypto_utils.constant_time_compare` helper wraps `hmac.compare_digest` and
should be preferred over direct string comparison.

When adding new authentication mechanisms:

- **Never** compare tokens with `==`. Use `constant_time_compare`.
- Keep keys out of the repository and load them from environment variables.
- Validate that network servers limit connection attempts to reduce brute force
  risks.

Unit tests (`tests/test_crypto_utils.py`) measure execution time for equal and
non-equal values to ensure the difference stays below a safe threshold.
