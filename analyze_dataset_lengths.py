"""Analyze sequence lengths in EasyTPP HuggingFace datasets."""

import numpy as np
from datasets import load_dataset

DATASETS = [
    "easytpp/retweet",
    "easytpp/stackoverflow",
    "easytpp/amazon",
    "easytpp/taxi",
]


def analyze_split(dataset_split, split_name):
    lengths = [len(item["time_since_start"]) for item in dataset_split]
    lengths = np.array(lengths)
    print(f"  {split_name}:")
    print(f"    Number of sequences: {len(lengths)}")
    print(f"    Average length:      {np.mean(lengths):.1f}")
    print(f"    Median length:       {np.median(lengths):.1f}")
    print(f"    Min length:          {np.min(lengths)}")
    print(f"    Max length:          {np.max(lengths)}")
    print(f"    Std length:          {np.std(lengths):.1f}")
    return lengths


def count_event_types(dataset):
    all_types = set()
    for split_name in dataset:
        for item in dataset[split_name]:
            all_types.update(item["type_event"])
    return len(all_types)


def main():
    for ds_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        dataset = load_dataset(ds_name)
        print(f"  Available splits: {list(dataset.keys())}")

        num_types = count_event_types(dataset)
        print(f"  Number of event types: {num_types}")

        all_lengths = []
        for split_name in dataset:
            lengths = analyze_split(dataset[split_name], split_name)
            all_lengths.extend(lengths)

        all_lengths = np.array(all_lengths)
        print(f"  Overall (all splits):")
        print(f"    Total sequences:     {len(all_lengths)}")
        print(f"    Average length:      {np.mean(all_lengths):.1f}")
        print(f"    Median length:       {np.median(all_lengths):.1f}")
        print(f"    Min length:          {np.min(all_lengths)}")
        print(f"    Max length:          {np.max(all_lengths)}")

        # Show distribution buckets
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        pvals = np.percentile(all_lengths, percentiles)
        print(f"    Percentiles:")
        for p, v in zip(percentiles, pvals):
            print(f"      {p:3d}th: {v:.0f}")


if __name__ == "__main__":
    main()
