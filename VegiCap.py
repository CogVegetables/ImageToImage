from collections import defaultdict, Counter
import mynn
import json
import numpy as np
import time
import mygrad as mg
import gensim
from gensim.models.keyedvectors import KeyedVectors
from sklearn.decomposition import TruncatedSVD
from mynn.layers.dense import dense
from mynn.optimizers.adam import Adam
import matplotlib.pyplot as plt


from mygrad.nnet.losses.margin_ranking_loss import margin_ranking_loss
from mygrad.nnet.initializers import glorot_normal

import pickle
import io
import requests
from PIL import Image

import re, string
punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))


path = r"./glove.6B.50d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(path, binary=False)

with open('resnet18_features.pkl', mode="rb") as open_file:
    resnet = pickle.load(open_file)


class COCOMappings:
    def __init__(self, glove):
        with open(r'./captions_train2014.json') as json_file:
            self.data = json.load(json_file)
            
        self.glove = glove
            
        self.caption_ids_to_captions = {cap['id']: cap['caption'] for cap in self.data['annotations']}
        self.image_ids_to_urls = {img['id']: img['coco_url'] for img in self.data['images']}
        
        self.img_ids = tuple(cap['image_id'] for cap in self.data['annotations'])
        self.caption_ids = tuple(cap['id'] for cap in self.data['annotations'])
        self.captions = tuple(cap['caption'] for cap in self.data['annotations'])
        
        self.caption_ids_to_embs = self.caption_id_to_emb()
        self.image_id_to_caption_id = self.generate_image_to_caption_id()
    
    def generate_image_to_caption_id(self):
        image_id_to_caption_ids_dict = {}
        
        for cap_dict in self.data['annotations']:
            if cap_dict['image_id'] in image_id_to_caption_ids_dict:
                image_id_to_caption_ids_dict[cap_dict['image_id']].append(cap_dict['id'])
            else:
                image_id_to_caption_ids_dict[cap_dict['image_id']] = [cap_dict['id']]
                
        return image_id_to_caption_ids_dict
        
    def tokenize(self, corpus):
        return punc_regex.sub('', corpus).lower().split()

    def to_df(self, captions):
        
        counter = Counter()
        for caption in captions:
            counter.update(set(self.tokenize(caption)))
        return dict(counter)
    
    
    def to_idf(self, captions):
        """ 
        Given the vocabulary, and the word-counts for each document, computes
        the inverse document frequency (IDF) for each term in the vocabulary.

        Parameters
        ----------
        vocab : Sequence[str]
            Ordered list of words that we care about.

        counters : Iterable[collections.Counter]
            The word -> count mapping for each document.

        Returns
        -------
        numpy.ndarray
            An array whose entries correspond to those in `vocab`, storing
            the IDF for each term `t`: 
                               log10(N / nt)
            Where `N` is the number of documents, and `nt` is the number of 
            documents in which the term `t` occurs.
        """
        vishnu = self.to_df(captions)
        return {word : np.log10(len(captions)/cnt + 1) for word, cnt in vishnu.items()}
    
    def normalize(self, array):
        #sqrroot(sum(vectorsquared))
        return (np.sum(array ** 2)) ** 0.5
        
        
    def caption_to_emb(self, caption):
        vishnu = sum(self.glove[word] * self.idfs[word] for word in self.tokenize(caption) if word in self.glove)
        return vishnu/self.normalize(vishnu)
    
    def caption_id_to_emb(self):
        idfs = self.to_idf(self.captions)
        self.idfs = idfs
        return {caption_id : self.caption_to_emb(self.caption_ids_to_captions[caption_id]) for caption_id in self.caption_ids}


def generate_triples(data, model, num_captions, trips_per_cap):
    triples = []
    for i in range(num_captions):
        img_id = None
        while img_id not in resnet: img_id = np.random.choice(data)
        cap_id = np.random.choice(model.image_id_to_caption_id[img_id])
        cap_emb = model.caption_ids_to_embs[cap_id]
        for n in range(trips_per_cap):
            bad_img_id = generate_bad_img(img_id, cap_emb, model, data)
            triples.append((img_id, cap_id, bad_img_id))
    triples = np.array(triples)
    np.random.shuffle(triples)
    return triples

def generate_bad_img(img_id, cap_emb, model, data):
    captions = []
    images = []
    for i in range(25):
        id_choice = img_id
        while id_choice == img_id or id_choice not in resnet: id_choice = np.random.choice(data)
        images.append(id_choice)
        captions.append(np.random.choice(model.image_id_to_caption_id[id_choice]))
    dots = np.zeros(25)
    for index, cap in enumerate(captions):
        emb = model.caption_ids_to_embs[cap]
        dots[index] = np.matmul(emb, cap_emb)
    max_index = np.argmax(dots)
    #bad_caption_id = captions[max_index]
    bad_img_id = images[max_index]
    return bad_img_id

def get_caption_embeddings(img_ids, dictionary):
    """
    Parameters
    ----------
    Sequence[int]
        N image IDs 
    dictionary : Dict[int, np.ndarray]
        img-ID -> shape-(50,) glove embedding vectors
    
    Returns
    -------
    shape-(N, 50)
        An array of the corresponding glove embedding vectors
    """
    vectors = np.zeros((len(img_ids), 50), dtype=np.float32)
    for n, _id in enumerate(img_ids):
        vectors[n] = dictionary[_id]
    return vectors

def get_image_embeddings(img_ids, resnet18_features):
    """
    Parameters
    ----------
    Sequence[int]
        N image IDs 
    resnet18_features : Dict[int, np.ndarray]
        img-ID -> shape-(512,) resnet vector
    
    Returns
    -------
    shape-(N, 512)
        An array of the corresponding resnet vectors
    """
    vectors = np.zeros((len(img_ids), 512), dtype=np.float32)
    for n, _id in enumerate(img_ids):
        vectors[n] = resnet18_features[_id]
    return vectors

class LinearEncoder:
    def __init__(self, d_input, d_output):
        """ This initializes all of the layers in our model, and sets them
        as attributes of the model.
        
        Parameters
        ----------
        d_input : int
            The size of the inputs.
            
        d_output : int
            The size of the outputs (i.e., the reduced dimensionality).
        """
        
        self.encoder = dense(d_input, d_output, weight_initializer=glorot_normal)
        
    def __call__(self, x):
        '''Passes data as input to our model, performing a "forward-pass".
        
        This allows us to conveniently initialize a model `m` and then send data through it
        to be classified by calling `m(x)`.
        
        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor], shape=(M, D_full)
            A batch of data consisting of M pieces of data,
            each with a dimentionality of D_full.
            
        Returns
        -------
        mygrad.Tensor, shape=(M, D_full)
            The model's prediction for each of the M pieces of data.
        '''
        
        return self.encoder(x) / mg.sqrt(mg.sum(x ** 2, axis = 1, keepdims = True))

    def save_model(self, path):
        """Path to .npz file where model parameters will be saved."""
        with open(path, "wb") as f:
            np.savez(f, *(x.data for x in self.parameters))

    def load_model(self, path):
        with open(path, "rb") as f:
            for param, (name, array) in zip(self.parameters, np.load(f).items()):
                param.data[:] = array
        
    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model.
        
        This can be accessed as an attribute, via `model.parameters` 
        
        Returns
        -------
        Tuple[Tensor, ...]
            A tuple containing all of the learnable parameters for our model """
        
        return self.encoder.parameters


def download_image(img_url):
    """ Fetches an image from the web.

    Parameters
    ----------
    img_url : string
        The url of the image to fetch.

    Returns
    -------
    PIL.Image
        The image."""

    response = requests.get(img_url)
    return Image.open(io.BytesIO(response.content))

def display_images(image_ids, mappings_object):
    for each in image_ids:
        download_image(mappings_object.image_ids_to_urls[each])
    return "Displaying top {} images".format(len(image_ids))


class ImageSemantics():
    
    def __init__(self, resnet):
        self.database = {}
        self.resnet = resnet
        
    def __repr__(self):
        return "Database of Image Semantics"
    
    def create_database(self, LinearEncoder, image_ids):
        for img_id in image_ids:
            self.database[img_id] = LinearEncoder(get_image_embeddings((img_id,), self.resnet))
            
    def query_database(self, caption, num_outs, mappings_object):
        caption_emb = mappings_object.caption_to_emb(caption)
        overlaps =  np.sum(caption_emb * np.array(list(self.database.values())), axis=1)
        overlaps = np.argsort(overlaps)
        k_imgs = overlaps[-1*num_outs:]
        display_images(k_imgs, mappings_object)
            
    def save_database(self, file_path):
        with open(file_path, mode='wb') as opened_file:
            pickle.dump(self.database, opened_file)
        return 'Database Saved'
            
    def load_database(self, file_path):
        with open (file_path, mode='rb') as opened_file:
            self.database = pickle.load(opened_file)
        return 'Database Loaded'