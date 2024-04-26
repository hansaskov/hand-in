from typing import Any
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import polars as pl

def plot_samples(Sequence, filename: str):
    plt.figure()
    plt.title("Sampling from the original HMM")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.plot(Sequence, label="Observations")
    plt.legend()
    plt.savefig(filename)

def normalize_sequences(sequences):
    normalized_sequences = []
    for seq in sequences:
        seq_norm = (seq-np.min(seq))/(np.max(seq)-np.min(seq))
        normalized_sequences.append(seq_norm)
    return normalized_sequences
 
def discritize_sequences(sequences, bins_amount):
    discretize_val = lambda val: int(val * bins_amount * 0.9999)
    discretize_seq = lambda seq: list(map(discretize_val, seq))
    return list(map(discretize_seq, sequences))

def split_sequences(sequences: np.ndarray[Any, Any], split_ratio=0.8):
    split_idx = int(split_ratio * len(sequences))
    train_sequences, test_sequences = sequences[:split_idx], sequences[split_idx:]
    return train_sequences.T, test_sequences.T


def run(n_components: int, m_bins: int):

    df = pl.read_csv("input.csv", columns=["Column6"])
    sequences = df.to_numpy()
    train_sequences, validation_sequences = split_sequences(sequences, split_ratio=0.8)

    print(train_sequences)

    # STEP 2. Normalize and discritize sequences
    normalized_sequences = normalize_sequences(train_sequences)
    discretized_sequences = discritize_sequences(normalized_sequences, m_bins)

    # STEP 3. Verify and Plot sequences
    print(train_sequences[0][:5])
    print(discretized_sequences[0][:5])
    plot_samples(train_sequences[0][:100], "sequence-continues.png")
    plot_samples(normalized_sequences[0][:100], "sequence-normalize.png")
    plot_samples(discretized_sequences[0][:100], "sequence-discrete.png")
    
    ## STEP 4: Train the HMM. 
    lengths = [len(sequence) for sequence in discretized_sequences]
    X = np.array(discretized_sequences).reshape(-1, 1)
 
    model = hmm.CategoricalHMM(n_components=n_components)
    model.fit(X, lengths)
        
    ## STEP 5: Verify and moniter convergence
    print("Converged?: ", model.monitor_.converged)
    print("Transimmions Matrix: \n", model.transmat_)
    print("Emmisions Matrix: \n", model.emissionprob_)

run(n_components=3, m_bins=3)