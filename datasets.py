import torch

import numpy as np

import gensim

from torch.utils.data import Dataset

import nltk

import utils

class ImdbReviewDatasetOneHot(Dataset):
    """
    ImdbReviewDatasetOneHot is a Dataset for processing IMDb movie reviews with one-hot encoded word vectors.

    Note: This method uses a lot of memory
    """
    def __init__(self, dataset, num_classes=2):
        """
        Initialize ImdbReviewDatasetOneHot.

        Parameters
        ----------
        dataset: pandas.DataFrame
            The input dataset containing 'review' and 'sentiment' columns.
        num_classes: int
            Number of classes for sentiment classification (default is 2).
        """
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download("punkt")

        super(ImdbReviewDatasetOneHot, self).__init__()
        self.dataset = dataset
        self.num_classes = num_classes

        parsed_sentences = list(map(lambda x: utils.preprocess(x), dataset["review"]))

        parsed_sentences = np.concatenate(parsed_sentences)
        #print(len(np.unique(parsed_sentences)))
        self.model = {}

        self.number_of_words = len(np.unique(parsed_sentences))
        for i, word in enumerate(np.unique(parsed_sentences)):
            # Only store the index except when retrieving word to save space
            self.model[word] = i



        
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Parameters
        ----------
        idx: int
            Index of the sample.

        Returns
        -------
        dict
            A dictionary containing 'word_vectors' (one-hot encoded word vectors),
            'label' (sentiment label), and 'natural_sentence' (original sentence in text).
        """
        
        words = self.dataset['review'][idx]
        sentiment_label = self.dataset['sentiment'][idx]

        sentiment_label = 1 if sentiment_label == "positive" else 0
        # Note: Punctuation is ignored.

        processed_words = utils.preprocess(words)

        word_vectors_neutral = []
        
        for word in processed_words:
            #Ignore words that are not in the model, all words should be in the model in this case
            if word in self.model:
                word__idx = self.model[word]
                word_vector = np.zeros(self.number_of_words)
                word_vector[word__idx] = 1
                word_vector_as_tensor = torch.as_tensor(word_vector)
                word_vectors_neutral.append(word_vector_as_tensor.clone())
            #else:
                #print("Missing word:", word)

        word_vectors_neutral = torch.stack(word_vectors_neutral)

        return {'word_vectors': word_vectors_neutral, "label": sentiment_label, "natural_sentence":  words}
    
class ToyReviewDataset(Dataset):
    """
    ToyReviewDataset is a Dataset for processing toy reviews with word vectors.
    """

    def __init__(self, dataset, pretrainfed_embedding, num_classes=2):
        """
        Initialize ToyReviewDataset.

        Parameters
        ----------
        dataset: pandas.DataFrame
            The input dataset containing 'sentence' and 'sentiment' columns.
        pretrainfed_embedding: bool
            Flag indicating whether to use pre-trained word embeddings or one-hot vectors.
        num_classes: int
            Number of classes for sentiment classification (default is 2).
        """
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download("punkt")

        super(ToyReviewDataset, self).__init__()
        self.dataset = dataset
        self.num_classes = num_classes

        # Train W2V model on dataset
        parsed_sentences = list(map(lambda x: utils.preprocess(x), dataset["sentence"]))

        parsed_sentences = np.concatenate(parsed_sentences)
        #print(len(np.unique(parsed_sentences)))
        if pretrainfed_embedding:
            self.model = gensim.models.KeyedVectors.load_word2vec_format('./embeddings/GoogleNews-vectors-negative300.bin', binary=True)
        else:
            self.model = {}

            number_of_words = len(np.unique(parsed_sentences))
            for i, word in enumerate(np.unique(parsed_sentences)):
                one_hot = np.zeros(number_of_words)
                one_hot[i] = 1
                self.model[word] = one_hot
            #print("Number of words:", number_of_words)


    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Parameters
        ----------
        idx: int
            Index of the sample.

        Returns
        -------
        dict
            A dictionary containing 'word_vectors' (word vectors),
                'label' (sentiment label), and 'natural_sentence' (original sentence in text).
        """

        words = self.dataset['sentence'][idx]
        sentiment_label = self.dataset['sentiment'][idx]

        sentiment_label = 1 if sentiment_label == " positive" else 0

        processed_words = utils.preprocess(words)

        word_vectors_neutral = []
        
        #print("processed word:", processed_words)
        for i, word in enumerate(processed_words):
            #Ignore words that are not in the model
            if word in self.model:
                word_vector = self.model[word].copy()

                word_vector_as_tensor = torch.as_tensor(word_vector)
                word_vectors_neutral.append(word_vector_as_tensor.clone())
            #else:
                #print("Missing word:", word)

        word_vectors_neutral = torch.stack(word_vectors_neutral)

        return {'word_vectors': word_vectors_neutral, "label": sentiment_label, "natural_sentence":  words}
    

class ImdbReviewsDataset(Dataset):
    """
    ImdbReviewDataset is a Dataset for processing IMDb movie reviews with Word2Vec 300 length vectors.
    """

    def __init__(self, dataset, num_classes=2):
        """
        Initialize ImdbReviewDataset.

        Parameters
        ----------
        dataset: pandas.DataFrame
            The input dataset containing 'review' and 'sentiment' columns.
        num_classes: int
            Number of classes for sentiment classification (default is 2).
        """
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download("punkt")

        super(ImdbReviewsDataset, self).__init__()
        self.dataset = dataset
        self.num_classes = num_classes

        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./embeddings/GoogleNews-vectors-negative300.bin', binary=True)

        # To Train W2V model on dataset:
        #parsed_sentences = list(map(lambda x: utils.preprocess(x), dataset["review"]))
        #self.word2vec_model = gensim.models.Word2Vec(sentences = parsed_sentences, vector_size=300, min_count=1, window=5, workers=4)
        #print(self.word2vec_model["hello"])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Parameters
        ----------
        idx: int
            Index of the sample.

        Returns
        -------
        dict
            A dictionary containing 'word_vectors' (word vectors),
                'label' (sentiment label), and 'natural_sentence' (original sentence in text).
        """
        words = self.dataset['review'][idx]
        sentiment_label = self.dataset['sentiment'][idx]

        sentiment_label = 1 if sentiment_label == "positive" else 0

        processed_words = utils.preprocess(words)

        word_vectors_neutral = []
        
        #print("processed word:", processed_words)
        for i, word in enumerate(processed_words):
            #Ignore words that are not in the model
            if word in self.word2vec_model:
                word_vector = self.word2vec_model[word].copy()

                word_vector_as_tensor = torch.as_tensor(word_vector)
                word_vectors_neutral.append(word_vector_as_tensor.clone())
            #else:
                #print("Missing word:", word)

        word_vectors_neutral = torch.stack(word_vectors_neutral)

        return {'word_vectors': word_vectors_neutral, "label": sentiment_label, "natural_sentence":  words}

class MnistDataset(Dataset):
    """
    MnistDataset is a Dataset for processing MNIST images with modified labels for binary classification. 
    This dataset modifies the labels for binary classification, creating pairs with positive and negative labels.
    """

    def __init__(self, dataset, num_classes=10):
        super(MnistDataset, self).__init__()
        self.dataset = dataset
        self.num_classes = num_classes

        self.switch=True
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Parameters
        ----------
        idx: int
            Index of the sample.

        Returns
        -------
        dict
            A dictionary containing 'image' (flattened image with label included),
                  'binary_label' (binary label indicating positive or negative label included in the image),
                  and 'label' (original label).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img, label = self.dataset[idx]
        #print(img)
        #print(label)

        image = img.clone()
        image = image.flatten()




        # Pick only one of negative and positive labels
        if np.random.random() > 0.5: # self.switch: # 

            #Positive label
            one_hot_label = torch.nn.functional.one_hot(
                torch.tensor(label), num_classes=self.num_classes
            )  
            image[0:self.num_classes] = one_hot_label


            return {'image': image, 'binary_label': 1.0, "label": label}
        else:
            #Negative label

            # Sample random false class; Sample in [0, classes-1], and add one if necessary to cover all classes but true label class
            random_class_label = np.random.choice(self.num_classes-1)
            if random_class_label >= label:
                random_class_label += 1
            
            one_hot_false_label = torch.nn.functional.one_hot(
                torch.tensor(random_class_label), num_classes=self.num_classes
            )

            image[0:self.num_classes] = one_hot_false_label

            return {'image': image, 'binary_label': 0.0, "label": label}
    
class MnistDatasetTriplet(Dataset):
    """
    MnistDatasetTriplet is a Dataset for processing MNIST images with modified labels for binary classification. Generates samples for positive, negative and neutral
    samples all within the same item.
    """
    def __init__(self, dataset,num_classes=10):
        #Num workers = 0 -> Only uses main process
        super(MnistDatasetTriplet, self).__init__()
        self.dataset = dataset
        self.num_classes = num_classes

        self.switch=True
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Parameters
        ----------
        idx: int
            Index of the sample.

        Returns
        -------
        dict
            A dictionary containing triplets of positive ("pos"), negative ("neg"), and neutral ("neutral") samples. 
            Each sample consist of a dictionary containing 'image' (flattened image with label included),
            'binary_label' (binary label indicating positive or negative label included in the image),
            and 'label' (original label).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #Probably same as what I did before.
        img, label = self.dataset[idx]
        #print(img)
        #print(label)

        image = img.clone()
        image = image.flatten()

        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(label), num_classes=self.num_classes
        )  
        #print(one_hot_label.shape)
        pos_sample=image.clone()

        pos_sample[0:self.num_classes] = one_hot_label

        pos = {'image': pos_sample, 'binary_label': 1.0, "label": label}
        random_class_label = np.random.choice(self.num_classes-1)
        if random_class_label >= label:
            random_class_label += 1
        
        one_hot_false_label = torch.nn.functional.one_hot(
            torch.tensor(random_class_label), num_classes=self.num_classes
        )
        neg_sample=image.clone()
        neg_sample[0:self.num_classes] = one_hot_false_label

        neg = {'image': neg_sample, 'binary_label': 0.0, "label": label}

        neutral_label = torch.zeros(self.num_classes) + 1/self.num_classes
        #print(one_hot_false_label.shape)

        neutral_sample=image.clone()

        neutral_sample[0:self.num_classes] = neutral_label

        neutral = {'image': neutral_sample, 'binary_label': 0.0, "label": label}
        return {"pos":pos,"neg":neg, "neutral":neutral}
        
        
