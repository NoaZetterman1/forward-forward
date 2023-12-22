import torch

def collate_text(batch):
    """
    Pads the beginning of each sentence in a batch of word-vectors sentences to align the length of all sentences in the batch.
    """

    longest_word_in_batch = 0

    # Find the longest word in given batch
    for item in batch:
        if longest_word_in_batch < len(item["word_vectors"]):
            longest_word_in_batch = len(item["word_vectors"])
    
    # Pad all words to that length
    batch_size = len(batch)
    word_vec_size = len(batch[0]["word_vectors"][0])

    #Init with zeros, could also be aligned with threshold value.
    words = torch.zeros((batch_size, longest_word_in_batch, word_vec_size))

    labels = torch.empty((batch_size))
    natural_sentences = []

    for i, item in enumerate(batch):
        words2 = item["word_vectors"]

        words[i, (longest_word_in_batch-len(words2)):,] = words2
        labels[i] = item["label"]
        natural_sentences.append(item["natural_sentence"])
    
    return {"word_vectors":words, "labels": labels, "natural_sentence":natural_sentences}

