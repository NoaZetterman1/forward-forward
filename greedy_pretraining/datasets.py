import torch

import numpy as np

import gensim

from torch.utils.data import Dataset

import nltk

import utils

class ImdbReviewsDataset(Dataset):
    

    def __init__(self, dataset, num_classes=2):
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download("punkt")

        super(ImdbReviewsDataset, self).__init__()
        self.dataset = dataset
        self.num_classes = num_classes

        # Train W2V model on dataset
        #parsed_sentences = list(map(lambda x: utils.preprocess(x), dataset["review"]))
        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./embeddings/GoogleNews-vectors-negative300.bin', binary=True)
        #self.word2vec_model = gensim.models.Word2Vec(sentences = parsed_sentences, vector_size=300, min_count=1, window=5, workers=4)
        #print(self.word2vec_model["hello"])

        
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if idx == 0:
            words = "very very good"
            sentiment_label = 1
        elif idx ==1:
            words = "really very bad"
            sentiment_label = 0
        elif idx == 2:
            words = "really not bad"
            sentiment_label = 1
        elif idx==3:
            words = "really not good"
            sentiment_label = 0
        elif idx==4:
            words ="really good"
            sentiment_label = 1
        elif idx==5:
            words = "really bad"
            sentiment_label = 0
        
        #words = self.dataset['review'][idx]
        #sentiment_label = self.dataset['sentiment'][idx]
        #print(words)
        #sentiment_label = 1 if sentiment_label == "positive" else 0
        # Note: Punctuation is ignored.
        #print("processing words:", words)
        # Convert words to w2v vectors
        processed_words = utils.preprocess(words)

        #print(words)

        word_vectors_neutral = []
        
        #print("processed word:", processed_words)
        for i, word in enumerate(processed_words):
            #Ignore words that are not in the model
            if word in self.word2vec_model:
                word_vector = self.word2vec_model[word].copy()

                word_vector_as_tensor = torch.as_tensor(word_vector)
                word_vectors_neutral.append(word_vector_as_tensor.clone())
            else:
                print("Missing word:", word)
        #print("Finished processing words:", word_vectors_neutral)
        word_vectors_neutral = torch.stack(word_vectors_neutral)
        #print(words)
        #print(word_vectors_neutral)
        return {'word_vectors': word_vectors_neutral, "label": sentiment_label, "natural_sentence":  words}

class MnistDataset(Dataset):
    def __init__(self, dataset, num_classes=10):
        #Num workers = 0 -> Only uses main process
        super(MnistDataset, self).__init__()
        self.dataset = dataset
        self.num_classes = num_classes

        self.switch=True
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img, label = self.dataset[idx]
        #print(img)
        #print(label)

        image = img.clone()
        image = image.flatten()




        # Pick only negative and positive labels
        if np.random.random() > 0.5: # self.switch: # 

            #Positive label
            one_hot_label = torch.nn.functional.one_hot(
                torch.tensor(label), num_classes=self.num_classes
            )  
            #print(one_hot_label.shape)
            image[0:self.num_classes] = one_hot_label


            return {'image': image, 'binary_label': 1.0, "label": label}
        else:
            #Negative label

            # Sample random number [0, classes-1], and ignore the correct class
            random_class_label = np.random.choice(self.num_classes-1)
            if random_class_label >= label:
                random_class_label += 1
            
            one_hot_false_label = torch.nn.functional.one_hot(
                torch.tensor(random_class_label), num_classes=self.num_classes
            )

            image[0:self.num_classes] = one_hot_false_label

            return {'image': image, 'binary_label': 0.0, "label": label}
    
class MnistDataset2(Dataset):
    def __init__(self, dataset,num_classes=10):
        #Num workers = 0 -> Only uses main process
        super(MnistDataset2, self).__init__()
        self.dataset = dataset
        self.num_classes = num_classes

        self.switch=True
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
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
        
        neutral_sample=image.clone()

        neutral_sample[0:self.num_classes] = neutral_label
        neutral = {'image': neutral_sample, 'binary_label': 0.0, "label": label}
        return {"pos":pos,"neg":neg, "neutral":neutral, "original": image, "label": label}
        
        
