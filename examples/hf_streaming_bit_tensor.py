"""Example of wrapping a HuggingFace streaming dataset with
``BitTensorStreamingDataset``.

The script downloads a tiny dataset without caching and streams it through the
wrapper, yielding encoded tensors on demand.
"""

from datasets import load_dataset

from bit_tensor_streaming_dataset import BitTensorStreamingDataset


def main() -> None:
    ds = load_dataset(
        "hf-internal-testing/fixtures_mixed",
        split="train",
        streaming=True,
    )
    stream_ds = BitTensorStreamingDataset(ds, virtual_batch_size=2)
    first_batch = stream_ds.get_virtual_batch(0)
    for idx, (inp, tgt) in enumerate(first_batch):
        print(f"sample {idx}:", inp.shape, tgt.shape)


if __name__ == "__main__":
    main()
