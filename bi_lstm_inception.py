from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, merge
from keras import backend as K 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import load_model
from PIL import ImageFile
from tensorflow.contrib.slim.python.slim.nets import inception
ImageFile.LOAD_TRUNCATED_IMAGES = True
import tensorflow as tf
import logging
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

import random
import time
import gc
import string
from tflearn.data_utils import to_categorical
import getopt
import sys
import os
import logging
import gensim
from gensim import utils
from nltk import word_tokenize, tokenize
from random import shuffle  #shuffle training data per epoch
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from nltk.probability import FreqDist
import datetime
from sklearn.metrics import classification_report, confusion_matrix

from os import listdir
from os.path import isfile, join
import shutil

import keras
from keras.layers import Dense, Input, Flatten
from keras.layers import Embedding, Reshape, Lambda
from keras.layers import MaxPooling1D, Embedding, Dropout, LSTM, Bidirectional, TimeDistributed, GlobalAveragePooling1D
from keras.models import Model
from keras import optimizers
from keras.engine.topology import Layer
from keras.models import load_model
from keras import backend as K 


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

tf.flags.DEFINE_float('learning_rate', 1e-4, 'learning rate to fine tune the last layer')
tf.flags.DEFINE_integer('img_width', 360, 'resized image width')
tf.flags.DEFINE_integer('img_height', 360, 'resized image height')
#tf.flags.DEFINE_boolean('horizontal_flip', False, 'whether to flip images horizontally')
tf.flags.DEFINE_boolean('dropout_flag', True, 'whether to dropout during training')
tf.flags.DEFINE_boolean('finetuning_last', False, 'whether to finetuning the last layer first')
tf.flags.DEFINE_string('train_data_dir', './training_1000_2000', 'training directory')
tf.flags.DEFINE_string('validation_data_dir', './validation_1000_2000', 'validation directory')
tf.flags.DEFINE_string('test_data_dir', './test_1000_2000', 'test directory')
tf.flags.DEFINE_integer('nb_train_samples', 24000, 'number of training samples')
tf.flags.DEFINE_integer('nb_validation_samples', 3000, 'number of validation samples')
tf.flags.DEFINE_integer('nb_test_samples', 2794, 'number of test samples')
tf.flags.DEFINE_integer('nb_classes', 6, 'number of classes')
tf.flags.DEFINE_string('saved_model_name', 'keras_inception.h5', 'the saved model file name')
tf.flags.DEFINE_integer('batch_size', 32, 'batch size')

tf.flags.DEFINE_integer("model_size", 350, "length of trimmed doc, number of sentences in a doc")
tf.flags.DEFINE_integer("sentence_length", 20, "number of words in a sentence")
tf.flags.DEFINE_integer("embedding_size", 50, "output size of the embedding layer")
tf.flags.DEFINE_integer("nb_epochs",50, "number of training epochs")
tf.flags.DEFINE_integer("frequency_bound", 20, "the lowest frequency for a word to be appeared in the dictionary")
tf.flags.DEFINE_integer("MAX_FILE_ID", 29794, "total number of instances")
tf.flags.DEFINE_integer("cell_size", 256, "number of neurons per layer")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "clip gradients to this norm.")
tf.flags.DEFINE_float("dropout_ratio", 0.5, "drop out probability, keep probability (1-dropout)")
tf.flags.DEFINE_integer("successive_decrease", 20, "number of successive decrease performance in validation dataset")
tf.flags.DEFINE_string("result", "result.txt", "the file name of result")
tf.flags.DEFINE_boolean("recompute_flag", False, "indicate whether to parse the document")
tf.flags.DEFINE_integer("hidden_size", 60, "hidden dimension after lstm output")
tf.flags.DEFINE_integer("data_flag", 5, "indicate which data set to use: 1 use 20432 data set; 2 use 29456 data set; 3 use 59451 data set")

FLAGS = tf.flags.FLAGS

if FLAGS.data_flag == 1:
    # using data set 20432
    data_dir = "../dataset_agg/sample_dataset/30_text"
    label_file = "../dataset_agg/sample_dataset/30_label"

elif FLAGS.data_flag == 5:
    data_dir ="./cleaned_dataset_29794"
    label_file = "./label_29794"
    #data_dir = "./sample_dataset/30_text"
    #label_file = "./sample_dataset//30_label"
elif FLAGS.data_flag == 6:
    train_dir = './data/ai_text/train'
    train_label = './data/ai_text/train_label.txt'
    
    vali_dir = './data/ai_text/vali'
    vali_label = './data/ai_text/vali_label.txt'
    
    test_dir = './data/ai_text/test'
    test_label = './data/ai_text/test_label.txt'
    
elif FLAGS.data_flag == 7:
    train_dir = './data/cl_text/train'
    train_label = './data/cl_text/train_label.txt'
    
    vali_dir = './data/cl_text/vali'
    vali_label = './data/cl_text/vali_label.txt'
    
    test_dir = './data/cl_text/test'
    test_label = './data/cl_text/test_label.txt'
    
elif FLAGS.data_flag == 8:
    train_dir = './data/lg_text/train'
    train_label = './data/lg_text/train_label.txt'
    
    vali_dir = './data/lg_text/vali'
    vali_label = './data/lg_text/vali_label.txt'
    
    test_dir = './data/lg_text/test'
    test_label = './data/lg_text/test_label.txt'

if FLAGS.nb_classes == 6:
    qualities = ["fa","ga","b","c","start","stub"]
elif FLAGS.nb_classes == 2:
    qualities = ['0', '1']

#train_data_dir = "./dataset_agg/sample_dataset/training_dataset"
#validation_data_dir = "./dataset_agg/sample_dataset/validation_dataset"
#train_data_dir = "./dataset_agg/sample_dataset/training_dataset"
#validation_data_dir = "./dataset_agg/sample_dataset/validation_dataset"
#test_data_dir = "./dataset_agg/sample_dataset/test_dataset"
#train_data_dir = "./training_1000_2000"
#validation_data_dir = "./validation_1000_2000"
#test_data_dir = './test_1000_2000'
#nb_train_samples = 24000
#nb_validation_samples = 3000
#nb_test_samples = 2794

def get_processor_version():
    version = str(tf.__version__)
    logger.info("tensorflow version: %s", version)
    logger.info("sentence level with word trim lstm version")
    logger.info("model_size: %d", FLAGS.model_size)
    logger.info("sentence_length: %d", FLAGS.sentence_length)
    logger.info("embedding_size: %d", FLAGS.embedding_size)
    logger.info("batch_size: %d", FLAGS.batch_size)
    logger.info("number of epochs: %d", FLAGS.nb_epochs)
    logger.info("frequency bound: %d", FLAGS.frequency_bound)
    logger.info("total number of files (instances): %d", FLAGS.MAX_FILE_ID)
    logger.info("lstm cell size: %d", FLAGS.cell_size)
    logger.info("learning rate %f", FLAGS.learning_rate)
    logger.info("max_grad_norm: %f", FLAGS.max_grad_norm)
    logger.info("dropout ratio: %f", FLAGS.dropout_ratio)
    logger.info("sucessive decrease: %d", FLAGS.successive_decrease)
    logger.info("name of result file: %s", FLAGS.result)
    logger.info("recompute_flag? %d", FLAGS.recompute_flag)
    logger.info("hidden dimension after reducing lstm output, %d", FLAGS.hidden_size)
    logger.info("data_flag? %d", FLAGS.data_flag)
    
    logger.info('resized img width: %d', FLAGS.img_width)
    logger.info('resized img height: %d', FLAGS.img_height)
    logger.info('training directory: %s', FLAGS.train_data_dir)
    logger.info('validation directory: %s', FLAGS.validation_data_dir)
    logger.info('test_data_dir: %s', FLAGS.test_data_dir)
    logger.info('number of training samples: %d', FLAGS.nb_train_samples)
    logger.info('number of validation samples: %d', FLAGS.nb_validation_samples)
    logger.info('number of test samples: %d', FLAGS.nb_test_samples)
    logger.info('number of classes: %d', FLAGS.nb_classes)
    logger.info('saved model file name: %s', FLAGS.saved_model_name)
    
    
    tensorflow_info = "tensorflow version: "+str(version)+"\n"
    version_info = "sentence level with word trim lstm version"+"\n"
    model_size = "model_size: "+str(FLAGS.model_size)+"\n"
    sentence_length = "sentence_length: "+str(FLAGS.sentence_length)+"\n"
    embedding_size = "embedding_size: " + str(FLAGS.embedding_size)+"\n"
    batch_size = "batch_size: "+str(FLAGS.batch_size)+"\n"
    nb_epochs = "number of epochs: "+str(FLAGS.nb_epochs)+"\n"
    frequency_bound = "frequency_bound: "+str(FLAGS.frequency_bound)+"\n"
    MAX_FILE_ID = "total number of files (instances): "+str(FLAGS.MAX_FILE_ID)+"\n"
    cell_size = "lstm cell size: "+str(FLAGS.cell_size)+"\n"
    learning_rate = "learning rate: "+str(FLAGS.learning_rate)+"\n"
    max_grad_norm = "max_grad_norm: "+str(FLAGS.max_grad_norm)+"\n"
    dropout_ratio = "dropout_ratio: "+str(FLAGS.dropout_ratio)+"\n" 
    successive_decrease = "successive decrease: "+str(FLAGS.successive_decrease)+"\n"
    result_name = "name of result file: "+FLAGS.result+"\n"
    recompute_flag = "recompute flag? "+str(FLAGS.recompute_flag)+"\n"
    hidden_size = "hidden size "+str(FLAGS.hidden_size)+"\n"
    data_flag = "data flag? "+str(FLAGS.data_flag)+"\n"
    
    resized_width = 'resized image width '+str(FLAGS.img_width)+'\n'
    resized_height = 'resized image height '+str(FLAGS.img_height)+'\n'
    training_directory = 'image training directory '+FLAGS.train_data_dir+'\n'
    validation_directory = 'image validation directory '+FLAGS.validation_data_dir+'\n'
    test_directory = 'image test directory '+FLAGS.test_data_dir+'\n'
    nb_train_samples = 'number of image training samples '+str(FLAGS.nb_train_samples)+'\n'
    nb_validation_samples = 'number of image validation samples '+str(FLAGS.nb_validation_samples)+'\n'
    nb_test_samples = 'number of image test samples '+str(FLAGS.nb_test_samples)+'\n'
    nb_classes = 'number of classes '+str(FLAGS.nb_classes)+'\n'
    saved_model = 'saved model file name: '+str(FLAGS.saved_model_name)+'\n'


    parameter_info = []
    parameter_info.append(tensorflow_info)
    parameter_info.append(version_info)
    parameter_info.append(model_size)
    parameter_info.append(sentence_length)
    parameter_info.append(embedding_size)
    parameter_info.append(batch_size)
    parameter_info.append(nb_epochs)
    parameter_info.append(frequency_bound)
    parameter_info.append(MAX_FILE_ID)
    parameter_info.append(cell_size)
    parameter_info.append(learning_rate)
    parameter_info.append(max_grad_norm)
    parameter_info.append(dropout_ratio)
    parameter_info.append(successive_decrease)
    parameter_info.append(result_name)
    parameter_info.append(recompute_flag)
    parameter_info.append(data_flag)
    parameter_info.append(hidden_size)
    
    parameter_info.append(resized_width)
    parameter_info.append(resized_height)
    parameter_info.append(training_directory)
    parameter_info.append(validation_directory)
    parameter_info.append(test_directory)
    parameter_info.append(nb_train_samples)
    parameter_info.append(nb_validation_samples)
    parameter_info.append(nb_test_samples)
    parameter_info.append(nb_classes)
    parameter_info.append(saved_model)
    
    return parameter_info

#load label file
def load_label (label_file):
    with open(label_file) as f:
        data = f.read()
        return data.splitlines()
        #return f.read().splitlines()

#load the content file
def load_content (file_name):
    with open(file_name, 'r') as f:
        line = f.read()
        #f.read().decode("utf-8").replace("\n", "<eos>")
        #return utils.to_unicode(line)
        #return line.encode('latin-1')
        #return str(line.encode('utf-16'))
        #a = str(line)
        #print(a)
        #return str(line.strip())
        return utils.to_unicode(line)
    
#tokenize doc into sentence level, then word level, and 
#return the word level tokenization of doc (X), total number of sentences for each document (nb_sentences)
#and number of words of each sentence for all sentence in a document (sentences_length_per_article), the number of total tokens for each doc
#(total_tokens). e.g., I have a dog. But I want a cat. 
#X is 'I', 'have', 'a', 'dog', '.', 'But', 'I', 'want', 'a', 'cat', '.'. nb_sentences is 2. sentences_length_per_article is [5, 6], total_tokens is 11 
def tokenization_train_validation_test():
    X = [] 
    #record number of sentences per article
    nb_sentences = [] 
    #record length of all sentences per article
    sentences_length_per_article = []
    #record total number of tokens for each document
    total_tokens = [] 
    
    train_file_list = [train_dir+'/'+f for f in listdir(train_dir) if isfile(join(train_dir, f))]
    sorted_train_file_list = sorted(train_file_list, key=lambda x: (x.split('/')[-1].replace('.json', '').split('.')[0], x.split('/')[-1].replace('.json', '').split('.')[1]))

    for i in range (0, len(sorted_train_file_list)):
        file_name = sorted_train_file_list[i]
        #print(file_name)
#         file_name = data_dir + '/' + str(i + 1)
        if i%200==0:
            logger.info("read the %d file ", i) 
        if os.path.isfile (file_name):
            document = load_content(file_name)
            sentences = tokenize.sent_tokenize(document)
            nb_sentences.append(len(sentences))
            temX = []
            tem_tokens= 0
            tem_sentences_length_per_article = []
            #temX = [word_tokenize(sentence) for sentence in sentences]
            for j in range(len(sentences)):
#                 if j < 3:    
#                     print (sentences[j])
                tem = word_tokenize(sentences[j])
                tem_tokens+=len(tem)
                tem_sentences_length_per_article.extend([len(tem)])
                temX.extend(tem)
            
            temX = [x.lower() for x in temX]
            X.append(temX)
            sentences_length_per_article.append(tem_sentences_length_per_article)
            total_tokens.append(tem_tokens)

    vali_file_list = [vali_dir+'/'+f for f in listdir(vali_dir) if isfile(join(vali_dir, f))]
    sorted_vali_file_list = sorted(vali_file_list, key=lambda x: (x.split('/')[-1].replace('.json', '').split('.')[0], x.split('/')[-1].replace('.json', '').split('.')[1]))

    for i in range (0, len(sorted_vali_file_list)):
        file_name = sorted_vali_file_list[i]
#         file_name = data_dir + '/' + str(i + 1)
        if i%200==0:
            logger.info("read the %d file ", i) 
        if os.path.isfile (file_name):
            document = load_content(file_name)
            sentences = tokenize.sent_tokenize(document)
            nb_sentences.append(len(sentences))
            temX = []
            tem_tokens= 0
            tem_sentences_length_per_article = []
            #temX = [word_tokenize(sentence) for sentence in sentences]
            for j in range(len(sentences)):
#                 if j < 3:    
#                     print (sentences[j])
                tem = word_tokenize(sentences[j])
                tem_tokens+=len(tem)
                tem_sentences_length_per_article.extend([len(tem)])
                temX.extend(tem)
            
            temX = [x.lower() for x in temX]
            X.append(temX)
            sentences_length_per_article.append(tem_sentences_length_per_article)
            total_tokens.append(tem_tokens)
            
    test_file_list = [test_dir+'/'+f for f in listdir(test_dir) if isfile(join(test_dir, f))]
    sorted_test_file_list = sorted(test_file_list, key=lambda x: (x.split('/')[-1].replace('.json', '').split('.')[0], x.split('/')[-1].replace('.json', '').split('.')[1]))

    for i in range (0, len(sorted_test_file_list)):
        file_name = sorted_test_file_list[i]
#         file_name = data_dir + '/' + str(i + 1)
        if i%200==0:
            logger.info("read the %d file ", i) 
        if os.path.isfile (file_name):
            document = load_content(file_name)
            sentences = tokenize.sent_tokenize(document)
            nb_sentences.append(len(sentences))
            temX = []
            tem_tokens= 0
            tem_sentences_length_per_article = []
            #temX = [word_tokenize(sentence) for sentence in sentences]
            for j in range(len(sentences)):
#                 if j < 3:    
#                     print (sentences[j])
                tem = word_tokenize(sentences[j])
                tem_tokens+=len(tem)
                tem_sentences_length_per_article.extend([len(tem)])
                temX.extend(tem)
            
            temX = [x.lower() for x in temX]
            X.append(temX)
            sentences_length_per_article.append(tem_sentences_length_per_article)
            total_tokens.append(tem_tokens)
            
    return X, nb_sentences, sentences_length_per_article, total_tokens

def tokenization():
    X = [] 
    #record number of sentences per article
    nb_sentences = [] 
    #record length of all sentences per article
    sentences_length_per_article = []
    #record total number of tokens for each document
    total_tokens = [] 

    for i in range (FLAGS.MAX_FILE_ID):
        file_name = data_dir + '/' + str(i + 1)
#         file_name = data_dir + '/' + str(i + 1)
        if i%2000==0:
            logger.info("read the %d file ", i) 
        if os.path.isfile (file_name):
            document = load_content(file_name)
            sentences = tokenize.sent_tokenize(document)
            nb_sentences.append(len(sentences))
            temX = []
            tem_tokens= 0
            tem_sentences_length_per_article = []
            #temX = [word_tokenize(sentence) for sentence in sentences]
            for j in range(len(sentences)):
#                 if j < 3:    
#                     print (sentences[j])
                tem = word_tokenize(sentences[j])
                tem_tokens+=len(tem)
                tem_sentences_length_per_article.extend([len(tem)])
                temX.extend(tem)
            
            temX = [x.lower() for x in temX]
            X.append(temX)
            sentences_length_per_article.append(tem_sentences_length_per_article)
            total_tokens.append(tem_tokens)
#         X.append (word_tokenize(load_content(file_name))) 
#     print (type(X))
#     X = np.asarray (X)
    return X, nb_sentences, sentences_length_per_article, total_tokens

#convert the nb_senetences into string format and write into file
def convert_array_to_string (data, output_file):
    res = ""
    for i in range(len(data)):
        res = res + str (data[i])
        if (i < len(data) - 1):
            res = res + '\t'
    
    with open(output_file, "w") as myfile:
        myfile.write(res)

#allocate a unique index for each unique word
def get_word2idx(datasets):
    word2idx = {}
    for data in datasets:
        for word in data:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    return word2idx

#get the frequency of each word 
def get_frequency(datasets, word2idx):
    vocab_size = len(word2idx)
    frequency = np.zeros(vocab_size)
    for data in datasets:
        for word in data:
            #if word in dictionary, update its corresponding frequency
            if word in word2idx:
                frequency[word2idx[word]]+=1
            
    return frequency


#words below the predefined frequency is treated as unknown words
def trim_frequency(datasets, frequency, word2idx, frequency_bound):
    #flatten datasets to make it more efficient
    datasets = [word for doc in datasets for word in doc]
    #only check the unique words
    datasets = set(datasets)
    
    new_datasets = []
#     print(len(frequency), len(new_datasets))
#     for data in new_datasets:
#         if frequency[word2idx[data]] < 20:
#             new_datasets.remove(data)
    for data in datasets:
        if frequency[word2idx[data]] >= FLAGS.frequency_bound:
            new_datasets.append(data)
    return new_datasets

def get_trimmed_word2idx1(datasets):
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 
    vocab = model.vocab.keys()
    
    word2idx = {}
    #one for padding, one for "UNK" unknown word 
    embedding_matrix = 2*np.random.random((len(datasets)+2, FLAGS.embedding_size))-1
    embedding_matrix[0] = np.zeros((1, FLAGS.embedding_size))
    word2idx["UNK"] = 1 
    datasets = set(datasets)
    i = 2
    count = 0
    logger.info("length of datasets: %d", len(datasets))
    for word in vocab:
        #numpy.ndarray
#         print(type(model.wv[word]))
        if word in datasets and word not in word2idx:
            word2idx[word] = i
#             embedding_matrix = np.concatenate((embedding_matrix, np.reshape(model[word],(1, FLAGS.embedding_size))))
           
            #shape of model.wv[word] is (embedding_size,1)
            embedding_matrix[i] = np.reshape(model[word],(1, FLAGS.embedding_size))
            i+=1
    logger.info("%d in the word2vec", len(word2idx)-1)         
    
    for word in datasets:
        if word not in word2idx:
            word2idx[word] = i
#             embedding_matrix = np.concatenate((embedding_matrix, np.random.random((1, FLAGS.embedding_size))))
#             embedding_matrix[i] = np.random.random((1, FLAGS.embedding_size))
            i+=1
            count+=1
    
    logger.info ("not in embedding %d", count)
    logger.info("%d overall vocabulary", len(word2idx))         
    return word2idx, embedding_matrix

def get_trimmed_word2idx(datasets):
    #load glove model 
    #get the name of glove model with corresponding embedding size
    glove_name = "./keras-bi-lstm/glove.6B."+str(FLAGS.embedding_size)+"d.txt"
#     glove_name = "/home/ailishen/data_partition/ailishen/data_set/wiki.en.vec"
#     glove_name = "wiki.en.vec"
    
    word2idx = {}
    #one for padding, one for "UNK" unknown word 
    embedding_matrix = 2*np.random.random((len(datasets)+2, FLAGS.embedding_size))-1
    embedding_matrix[0] = np.zeros((1, FLAGS.embedding_size))
    word2idx["UNK"] = 1 
    datasets = set(datasets)
    i = 2
    count = 0
    logger.info("length of datasets: %d", len(datasets))
    
    #build word-vector dictionary
    logger.info("Loading Glove Model")
    file = open(glove_name, 'r')    
    for line in file:
        splitLine = line.split()
        word = splitLine[0]
        if word in datasets and word not in word2idx:
            word2idx[word] = i
            embedding_matrix[i] = [float(val) for val in splitLine[1:]]
            i+=1    
    file.close()
    
    logger.info("%d in the word2vec", len(word2idx)-1)         
    
    for word in datasets:
        if word not in word2idx:
            word2idx[word] = i
#             embedding_matrix = np.concatenate((embedding_matrix, np.random.random((1, FLAGS.embedding_size))))
#             embedding_matrix[i] = np.random.random((1, FLAGS.embedding_size))
            i+=1
            count+=1
    
    logger.info ("not in embedding %d", count)
    logger.info("%d overall vocabulary", len(word2idx))  

    return word2idx, embedding_matrix
#substitute each word in the doc with its corresponding index in the dictionary
#word with frequency below a specified value is treated as unknown word, and the index of UNK is
#assigned to the word

########## to do on 29th to vectorize data according to the pre-trained embedding  
def vectorize_data(docs, word2index, sentences_length_per_article):
    X = []
    i = 0 #the index of document
    trimmed_sentences_length_per_article = []
    while i < len(docs):
        doc = docs[i]
        x = np.zeros((FLAGS.model_size, FLAGS.sentence_length))
        total = 0
        sentence_number = 0
        tem_trimmed = np.ones([FLAGS.model_size]) #the default set as one in case of divided by 0
        for j in sentences_length_per_article[i]:
            if j < FLAGS.sentence_length:
                tem_trimmed[sentence_number] = j+1 #all sentence length add 1
            else:
                #trim long sentence into specified length
                tem_trimmed[sentence_number] = FLAGS.sentence_length+1#if sentence length exceeds sentence_length, set it as sentence_length+1
            
            for tem in range(int(tem_trimmed[sentence_number]-1)):
                tem_idx = 0
                word = doc[total+tem]
                #if word is in the dictionary explicitly, get the word index from the dictionary
                #otherwise, get the Unknown word index
                if word in word2index:
                    tem_idx = word2index[word]
                else:
                    tem_idx = word2index["UNK"]
                x[sentence_number][tem]=tem_idx

            total+=j
            sentence_number+=1
            if(sentence_number == FLAGS.model_size):
                break
        X.append(x)
#         print(tem_trimmed)
        trimmed_sentences_length_per_article.append(tem_trimmed) 
        i+=1
    X = np.array(X)
    trimmed_sentences_length_per_article = np.array(trimmed_sentences_length_per_article)
    return X, trimmed_sentences_length_per_article
 
#record per sentence length for all sentences of all docs
def write_sentence_length(sentences_length_all_articles, output_file):
    with open(output_file, "w") as myfile:
        for sentences_per_article in range(len(sentences_length_all_articles)):
            res = ""
            for i in range(len(sentences_length_all_articles[sentences_per_article])):
                res = res+str(sentences_length_all_articles[sentences_per_article][i])
                if(i < len(sentences_length_all_articles[sentences_per_article])-1):
                     res = res+'\t'
                else:
                    res = res+'\n'
            myfile.write(res)
            
#load per sentence length for all sentences of all docs
def load_sentence_length(input_file):
    sentences=[]
    myfile = open(input_file, "r")
    lines = myfile.readlines()
    for line in lines:
#         print("world", line)
        sentence_per_article= []
        length = line.split('\t')
#         print("hello world", length)
        for i in range(len(length)):
            sentence_per_article.extend([int(float(length[i]))])
        sentences.append(sentence_per_article)
    myfile.close()  
    sentences = np.asarray(sentences) 
    return sentences
#record the embedding vector
def write_embedding(embedding_matrix, output_file):
    with open(output_file, "w") as myfile:
        for embedding in embedding_matrix:
            res = ""
            for i in range(FLAGS.embedding_size):
                res = res+str(embedding[i])
                if i < FLAGS.embedding_size-1:
                    res = res+'\t'
            res = res+'\n'
            myfile.write(res)
            
#record vectorized data and vocabulary size
def write_vectorize_data(datasets, voca_size, output_file):
    with open(output_file, "w") as myfile:
        #write vocabulary size into the file
        myfile.write(str(voca_size))
        myfile.write('\n')
        
        for data in datasets:
            res = ""
            for i in range(len(data)):
#                 print('hadahdfa', data)
                for j in range(len(data[i])):
                    res = res+str(data[i][j])
                    if(j < len(data[i])-1):
                        res = res+'\t'
                if (i <len(data)-1):
                    res = res+'\t'
            res = res+'\n'
            myfile.write(res)

#load embedding vector
def load_embedding(input_file):
    embedding_matrix = []
    with open(input_file, "r") as f:
        embeddings = f.read().splitlines()
        for embedding in embeddings:
            tem_embedding = embedding.split('\t')
            embedding_matrix.append(tem_embedding)
    return np.asarray(embedding_matrix, dtype='float32')
           
#load vectorized data and vocabulary size
def load_vectorize_data(input_file):
    with open(input_file, "r") as myfile:
        line = myfile.readline()
        voca_size = int(line)
        X = []
        j = 0 
        line = myfile.readline()
        while line:
            if(j % 2000 == 0):
                logger.info("load the %d file", j)
            
            #temX represent one document
            temX = []
            vectorized = line.split('\t')
            temx = []
            
            for k in range(len(vectorized)):
                tem = int(float(vectorized[k]))
                temx.extend([tem])
                if((k+1)%FLAGS.sentence_length==0):
                    temX.append(temx)
                    temx = []
            j+=1

            X.append(temX)
            line = myfile.readline()
    X = np.asarray(X)
    return voca_size, X

#load trimmed number of sentences each doc
def load_trimmed_nb_sentences (input_file):
    with open (input_file) as f:
        trimmed_nb_sentences = f.read().split('\t')
        return np.asarray(trimmed_nb_sentences) 
    
def read_index_file(file_name):
    with open(file_name, 'r') as f:
        data = f.read()
        data = data.splitlines()
        for i in range(0, len(data)):
            data[i] = int(data[i])
            
        return data

def get_file_index(filename_list):
    index_list = []
    for i in range(0, len(filename_list)):
        filename = filename_list[i]
        filename = filename.split('/')[-1].replace('.jpg', '')
        index_list.append(int(filename)-1)
    return index_list

'''def custom_generator(generator, dir, text_input, sentence_trimmed_input, trimmed_nb_sentences_input, shuffle_flag, flag=None):
        gen = generator.flow_from_directory(dir, target_size = (FLAGS.img_height, FLAGS.img_width),
                                              class_mode = 'categorical',
                                              batch_size = FLAGS.batch_size,
                                              shuffle=shuffle_flag)

        while True:
            X_triple = gen.next()
            filename = X_triple[2]
            index_list = get_file_index(filename)
            batch_text_X = text_input[index_list]
            #sentence length for each sentence in one article
            batch_sentence_trimmed = sentence_trimmed_input[index_list]
            #nb of sentences in one article
            batch_trimmed_nb_sentences = trimmed_nb_sentences_input[index_list]
            yield [batch_text_X, batch_sentence_trimmed, batch_trimmed_nb_sentences, X_triple[0]], X_triple[1]'''

def custom_generator(generator, dir, text_input, shuffle_flag, flag=None):
        gen = generator.flow_from_directory(dir, target_size = (FLAGS.img_height, FLAGS.img_width),
                                              class_mode = 'categorical',
                                              batch_size = FLAGS.batch_size,
                                              shuffle=shuffle_flag)

        while True:
            X_triple = gen.next()
            filename = X_triple[2]
            index_list = get_file_index(filename)
            batch_text_X = text_input[index_list]
            yield [X_triple[0], batch_text_X], X_triple[1]
            
def average_sentence(x):    
    import tensorflow as tf
    FLAGS = tf.flags.FLAGS 
    actual_x = x[0]
    trimmed = x[1]
    #print(actual_x.shape)
    #print(trimmed.shape)
    reduced_sum = tf.reduce_sum(actual_x, axis=2)
    #print(reduced_sum.shape)
    average = reduced_sum/tf.expand_dims(trimmed, -1)
    print('average shape before return')
    print(average.shape)
    return average

def average_sentence_output_shape(input_shape):
    import tensorflow as tf
    FLAGS = tf.flags.FLAGS
    shape = list(input_shape)
    print('haha')
    print(shape)
    return (None, FLAGS.model_size, FLAGS.embedding_size)

def average_doc(x):
    import tensorflow as tf
    FLAGS = tf.flags.FLAGS
    actual_doc = x[0]
    actual_length = x[1]
    reduced_sum = tf.reduce_sum(actual_doc, axis=1)
    average = reduced_sum/actual_length
    print('doc level shape before return')
    print(average.shape)
    return average

def average_doc_output_shape(input_shape):
    import tensorflow as tf
    FLAGS = tf.flags.FLAGS
    shape = list(input_shape)
    print('doc level haha')
    print(shape)
    return (None, 2*FLAGS.cell_size)


if __name__ == '__main__':
    para_info = get_processor_version()
    result_file = open(FLAGS.result, 'w+')
    result_file.truncate()
    result_file.write("start time: "+str(datetime.datetime.now())+"\n")
    result_file.writelines(para_info)
    
    logger.info('Read labels')
    print(label_file)
    Y = load_label(label_file)

    for i in range(len(Y)):
        Y[i] = qualities.index(Y[i].lower())

    Y = Y[:FLAGS.MAX_FILE_ID]
    '''nb_fa_total = Y.count(0)
    nb_ga_total = Y.count(1)
    nb_b_total = Y.count(2)
    nb_c_total = Y.count(3)
    nb_start_total = Y.count(4)
    nb_stub_total = Y.count(5)
    logger.info("total number of fa: %d,  ga: %d, b class: %d, c class: %d, start: %d, stub: %d", nb_fa_total, nb_ga_total, nb_b_total, nb_c_total, nb_start_total, nb_stub_total)
    result_file.write("total number of fa: "+str(nb_fa_total)+" ga: "+str(nb_ga_total)+" b class: "+str(nb_b_total)+" c class: "+str(nb_c_total)+" start: "+str(nb_start_total)+" stub: "+str(nb_stub_total)+"\n")'''
    Y = np.array(Y)
    
    string_sentences = "sentences_"+str(FLAGS.model_size)+".txt"
    string_total_tokens = "total_tokens_"+str(FLAGS.model_size)+".txt"
    string_sentence_length_per_article = "sentence_length_per_article_"+str(FLAGS.model_size)+".txt"
    string_vectorized = "vectorized_"+str(FLAGS.model_size)+".txt"
    string_trimmed_sentences = "trimmed_sentences_"+str(FLAGS.model_size)+".txt"
    string_trimmed_sentence_length_per_article = "trimmed_sentence_length_per_article_"+str(FLAGS.model_size)+".txt"
    if FLAGS.recompute_flag == True:
        logger.info('Read content')
        X, nb_sentences, sentences_length_per_article, total_tokens = tokenization()
#         sorted_length = sorted(nb_sentences)
        convert_array_to_string(nb_sentences, string_sentences)
        convert_array_to_string(total_tokens, string_total_tokens)
        write_sentence_length(sentences_length_per_article, string_sentence_length_per_article)
        logger.info("sentence length written done")

        word2idx = get_word2idx(X)
        logger.info("word2index")
        #idx2word = dict(zip(word2idx.values(), word2idx.keys()))
        frequency = get_frequency(X, word2idx)
        logger.info("frequency")
        datasets = trim_frequency(X, frequency, word2idx, FLAGS.frequency_bound)
        logger.info("trimmed_frequency")
        word2idx, embedding_matrix = get_trimmed_word2idx(datasets)
        vocab_size = len(word2idx)+1#plus 1 because the padding is not in the dictionary
        
        write_embedding(embedding_matrix, "embedding.txt")
        #sentences_length_per_article is the trimmed ones
        X, sentences_length_per_article = vectorize_data(X, word2idx, sentences_length_per_article)
#         print(X[0]) 
        write_vectorize_data(X, vocab_size, string_vectorized)
        
                
        #trim number of sentences per article to model size
        trimmed_nb_sentences = []
        for i in range(len(nb_sentences)):
            if nb_sentences[i] > FLAGS.model_size:
                trimmed_nb_sentences.extend([FLAGS.model_size])
            else:
                trimmed_nb_sentences.extend([nb_sentences[i]])
        trimmed_nb_sentences = np.asarray(trimmed_nb_sentences)        
        convert_array_to_string(trimmed_nb_sentences, string_trimmed_sentences)
        write_sentence_length(sentences_length_per_article, string_trimmed_sentence_length_per_article)
        logger.info("vectorized data written done")
    else:
        sentences_length_per_article = load_sentence_length(string_trimmed_sentence_length_per_article)
        embedding_matrix = load_embedding("embedding.txt")
        vocab_size, X = load_vectorize_data(string_vectorized)
        trimmed_nb_sentences = load_trimmed_nb_sentences(string_trimmed_sentences)
        logger.info("loading sentence length, vocabulary size and vectorized data done")
        logger.info(type(X))
    
        
    logger.info("vocabulary size %d", vocab_size)
    result_file.write("vocabulary size: "+str(vocab_size)+"\n")

    logger.info('Finish reading data')
    result_file.write("Finish reading data \n")
    #FLAGS.model_size=35
    
    Y = to_categorical(np.asarray(Y), nb_classes=FLAGS.nb_classes)
    
    #static model
    text_input = Input(shape=(FLAGS.model_size, FLAGS.sentence_length))
    embedding_layer = Embedding(vocab_size, FLAGS.embedding_size, weights=[embedding_matrix], mask_zero=False, input_length=FLAGS.sentence_length*FLAGS.model_size, trainable=True, input_shape=(FLAGS.model_size, FLAGS.sentence_length))(text_input)
    reshaped_word_embedding = Reshape(target_shape=(FLAGS.sentence_length, FLAGS.embedding_size*FLAGS.model_size))(embedding_layer)
    sentence_embedding = keras.layers.GlobalAveragePooling1D()(reshaped_word_embedding)
    reshaped_sentence_embedding = Reshape(target_shape=(FLAGS.model_size, FLAGS.embedding_size))(sentence_embedding)
    dropped_sentence_embedding = Dropout(0.5)(reshaped_sentence_embedding)
    lstm_output_sequence = Bidirectional(LSTM(FLAGS.cell_size, return_sequences=True))(dropped_sentence_embedding)
    lstm_output = keras.layers.GlobalAvgPool1D()(lstm_output_sequence)
    dropped_lstm_output = Dropout(0.5)(lstm_output)
    
    #dynamic model
    '''sentence_input = Input(shape=(FLAGS.model_size, FLAGS.sentence_length,), dtype='int32')
    sentence_trimmed_input = Input(shape=(FLAGS.model_size,), dtype='float32')
    embedding_layer = Embedding(vocab_size, FLAGS.embedding_size, weights=[embedding_matrix], mask_zero=True, input_length=FLAGS.model_size*FLAGS.sentence_length, trainable=True, input_shape=(FLAGS.model_size, FLAGS.sentence_length,))(sentence_input)
    averaged_sentence = Lambda(average_sentence, output_shape=average_sentence_output_shape)([embedding_layer, sentence_trimmed_input])
    
    dropped_sentence_embedding = Dropout(0.5)(averaged_sentence)
    lstm_output_sequence = Bidirectional(LSTM(FLAGS.cell_size, return_sequences=True))(dropped_sentence_embedding)
    trimmed_nb_sentences_input = Input(shape=(1,), dtype='float32')
    lstm_output = Lambda(average_doc, output_shape=average_doc_output_shape)([lstm_output_sequence, trimmed_nb_sentences_input])
    dropped_lstm_output = Dropout(0.5)(lstm_output)'''

    image_input = Input(shape=(FLAGS.img_width, FLAGS.img_height, 3))
    base_model = applications.InceptionV3(weights="imagenet", include_top=False, input_shape=(FLAGS.img_width, FLAGS.img_height, 3))
    #penultimate_layer = base_model.output
    penultimate_layer = base_model(image_input)
    #use dropout of ratio 0.5
    dropped_penultimate_layer = Dropout(0.5)(penultimate_layer)
    inception_features = GlobalAveragePooling2D(name='avg_pool')(dropped_penultimate_layer)
    
    
    joint = merge([dropped_lstm_output, inception_features], mode='concat', concat_axis=1)
    joint = Dense(512, activation='relu')(joint)
    predictions = Dense(FLAGS.nb_classes, activation='softmax', name='predictions')(joint)
    
    # this is the model we will train
    #model_final = Model(inputs=[sentence_input, sentence_trimmed_input, trimmed_nb_sentences_input, image_input], outputs=predictions)
    model_final = Model(inputs=[image_input, text_input], outputs=predictions)
       
    # compile the model (should be done *after* setting layers to non-trainable)
    model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=FLAGS.learning_rate), metrics=["accuracy"])
    #model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=FLAGS.learning_rate), metrics=["accuracy"])
    
    print(model_final.summary())
    
    train_datagen = ImageDataGenerator(fill_mode="nearest", zoom_range=0.1, \
                                       width_shift_range=0.1, height_shift_range=0.1, rotation_range=0)
    
    vali_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

#     train_generator = custom_generator(train_datagen, FLAGS.train_data_dir, X, sentences_length_per_article, trimmed_nb_sentences, shuffle_flag=True, flag='train')
#     validation_generator = custom_generator(vali_datagen, FLAGS.validation_data_dir, X, sentences_length_per_article, trimmed_nb_sentences, shuffle_flag=False)
#     test_generator = custom_generator(test_datagen, FLAGS.test_data_dir, X, sentences_length_per_article, trimmed_nb_sentences, shuffle_flag=False)
    train_generator = custom_generator(train_datagen, FLAGS.train_data_dir, X, shuffle_flag=True, flag='train')
    validation_generator = custom_generator(vali_datagen, FLAGS.validation_data_dir, X, shuffle_flag=False)
    test_generator = custom_generator(test_datagen, FLAGS.test_data_dir, X, shuffle_flag=False)
    test_generator1 = test_datagen.flow_from_directory(FLAGS.test_data_dir, target_size=(FLAGS.img_height, FLAGS.img_width), \
                                class_mode='categorical', shuffle=False)
    
    # Save the model according to the conditions  
    checkpoint = ModelCheckpoint(FLAGS.saved_model_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=2, mode='auto')
    
    #K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)))
    
    # Train the model 
    model_final.fit_generator(train_generator, steps_per_epoch=FLAGS.nb_train_samples/float(FLAGS.batch_size), epochs=FLAGS.nb_epochs, max_queue_size=10,\
                              validation_data=validation_generator, validation_steps=np.ceil(FLAGS.nb_validation_samples/float(FLAGS.batch_size)), callbacks=[checkpoint, early])
    
    del model_final
    print('***************************************')
    print('start to predict on the test set using saved model best on the validation set')
    model_final = load_model(FLAGS.saved_model_name)
    result1 = model_final.predict_generator(test_generator, steps=np.ceil(FLAGS.nb_test_samples/float(FLAGS.batch_size)), use_multiprocessing=False, verbose=1)
    result2 = model_final.evaluate_generator(test_generator, steps=np.ceil(FLAGS.nb_test_samples/float(FLAGS.batch_size)), use_multiprocessing=False)
    
    print('result2')
    print(result2)
    y_pred = np.argmax(result1, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(test_generator1.classes, y_pred))
    print('Classification Report')
    target_names = ['0', '1', '2', '3', '4', '5']
    print(classification_report(test_generator1.classes, y_pred, target_names=target_names))
