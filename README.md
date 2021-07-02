# NLP_Word_Embedding_Techniques_for_TextAnalysis
Word Embedding Technique using Embedding Layer in Keras <hr><br><br>

### INTRODUCTION<hr>
Let’s talk about text data<hr> <br><br>

* It doesn’t matter if you’re spending that lazy Sunday watching shows on Netflix, or filling up that last-minute application to your favorite grad school, you’re surrounded by data. In this article, we will focus on one of the most common forms of data, i.e., “text.”

* News articles, emails, chats, and even blogs such as this one are full of text-based data. Now, with the amount of textual data readily available at our disposal, wouldn’t it make sense to develop a deeper understanding of this data and gain potential revolutionary insights?

### Natural Language Processing, Text Analysis, and its importance<hr><br>

* That’s where Natural Language Processing comes into the picture. It combines our knowledge of linguistics, machine learning, computer science to analyze huge chunks of natural language data, to create useful applications for our society.

* NLP finds applications in several fields. Here we will lay our focus on Text Analysis and ‘word embedding’ in particular.

* Companies use a technique called ‘Text Analysis’ to classify essential emails and texts, understand the general customer sentiment without manually reading reviews and feedback. It helps companies save thousands of hours of human resources, money, and helps automate several mundane processes.

### Word embedding techniques<hr><br>

* Word embedding and its implementations will be the highlight of this article.
* Word embedding helps capture the semantic, syntactic context or a word/term and helps understand how similar/dissimilar it is to other terms in an article, blog, etc.
* Word embedding implements language modeling and feature extraction based techniques to map a word to vectors of real numbers. Some of the popular word embedding methods are:
    1 - Binary Encoding.
    2 - TF Encoding.
    3 - TF-IDF Encoding.
    4 - Latent Semantic Analysis Encoding.
    5 - Word2Vec Embedding.

We will cover TF-IDF Encoding, Word2Vec Embedding in-depth with explanations, diagrams, and code.


## TF-IDF<hr><br>

### Introduction<br><br>
* TF-IDF, short for term frequency-inverse document frequency, can break a word into two parts: TF and IDF.

* TF is the term abbreviation of Term Frequency, defined as the total number of times a term occurs in a document. TF is calculated using the number of times the term occurs in a document divided by the total number of terms. The method is quite intuitive. The more a term occurs in a document, the more significance this term holds against the document.

* However, this is not always the case, depending on the content of documents. From the term frequency that we calculate, we would probably see that terms like “a,” “of,” “the” have the largest value. These words won’t help our analysis much since the word “the” is so common that every article has it. And having these grammatical purpose words is very likely to affect our analysis outcome in the end. One approach to handle this problem is simply removing them.

* We can use the Python package nltk to download all stop words in English.<hr><br><br>
  <mark># Load library
  from nltk.corpus import stopwords
  # You will have to download the set of stop words the first time
  import nltk
  nltk.download('stopwords')
  # Load stop words
  stop_words = stopwords.words('english')<hr><br><br><br>
  
* From now on, we have just successfully filtered out all stop words, and the remaining are those which hold actual meaning. However, we could probably run into another problem. Suppose we are analyzing news about cats in Canada, and we have calculated the term frequency. In one of these articles, we see words like “Canada,” “animal,” “food,” “cats,” having the same term frequency values. Does it mean these words hold the same amount of importance against the article? The answer is probably No. But the question is, how do we solve this problem? We use IDF.

 * Since TF tends to emphasize the wrong term sometimes, IDF is introduced to balance the term weight. IDF, short for inverse document frequency, defined as how frequently a term occurs in the entire document. It is used to balance the weight of terms that occur in the entire document set. In other words, IDF reduces the weight of terms that occur frequently and increases the weight of terms that occur infrequently.

  * IDF is calculated using the number of documents containing the term divided by the total number of documents before taking the logarithm. If the term often appears in the entire document set, the IDF result will be close to 0. Otherwise, it will be increasing towards positive infinity.

  * To get the final TF-IDF score, we need to multiply the results of TF and IDF. The larger the TF-IDF score is, the more relevant the term is in the documents. From the result, we can see that TF-IDF is proportional to the number of times a word appears in an article, and inverse proportional to the number of times this word appears in the entire domain of articles.

  * The mathematical equations for calculating TF-IDF (using the word ‘w’ as an example):

    TF(w)=Frequency of w occurs in the document / the total number of words
    IDF(w)=log(the number of documents containing the term)/ the total number of documents(+1) .
          (Note: the reason to add one in the denominator is to avoid division by zero)
          TF-IDF(w)=TF(w)*IDF(w)

  ### Code tutorial<hr><br>

In this part, we want to show how TF-IDF works, so we write code by ourselves instead of using the function offered by Spark. The code outputs the five words with the highest TF-IDF score in each article.

<code>import pandas as pd
import os
import re
import sys
from nltk.corpus import stopwords
import nltk
import math
# Download stop words
nltk.download('stopwords')
stop_words = stopwords.words('english')
class TFIDF:
    def __init__(self, filename):
        self.filename = filename
        self.text = self.tokenize(filename)
        self.text_df = self.load_df()
# tokenize raw text into words and then remove stop words
    def tokenize(self, filename):
        if not os.path.exists(filename):
            return []
        # read the file
        with open(filename, 'r', encoding="utf-8", errors='ignore') as fi:
            raw_text = fi.read()
            regex = r'\w+'
            text = re.findall(regex, raw_text)
        # take lowercase and remove stop words
        text = [word.lower() for word in text]
        text = [word for word in text if not word in stop_words]
        return text
# turn list of words into pandas dataframe
    def load_df(self):
        text_df = pd.DataFrame(self.text, columns=['word'])
        text_df = text_df.groupby('word').size().reset_index()
        text_df = text_df.rename(columns={0: 'count'})
return text_df
# calculate term frequency
    def tf(self):
        size = len(self.text)
        self.text_df['tf'] = self.text_df['count'] / size
# calculate inverse document frequency
    def idf(self, all_text, document_size):
        self.text_df['idf'] = self.text_df['word'].apply(lambda word: self.count_idf(word, all_text, document_size))
# helper for idf
    def count_idf(self, word, all_text, document_size):
        count = 0
        for text in all_text:
            if word in text:
                count = count + 1
        return math.log(document_size / (count + 1))
# caculate tfidf together
    def tf_idf(self):
        self.text_df['tfidf'] = self.text_df['tf'] * self.text_df['idf']
        self.text_df = self.text_df.round(3)
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Only pass one directory name")
        exit(1)
# read all filenames of the directory
    directory = sys.argv[1]
    filenames = []
    for (dirpath, dirnames, files) in os.walk(directory):
        filenames.extend(files)
# construct all TFIDF object for all files
    tfidfs = []
    document_size = len(filenames)
    # calculate tf
    for filename in filenames:
        tfidf = TFIDF(directory + '/' + filename)
        tfidf.tf()
        tfidfs.append(tfidf)
# calculate idf
    all_text = [tfidf.text for tfidf in tfidfs]
    for i, tfidf in enumerate(tfidfs):
        tfidf.idf(all_text, document_size)
        tfidf.tf_idf()
        result_df = tfidf.text_df.sort_values('tfidf', ascending=False).head(5)  # sort and take top 5
        result_words = []
        for index, row in result_df.iterrows():
            result_words.append((row.word, row.tfidf))
        print(tfidf.filename, result_words)<code><hr><br><br><br>
  
  
* The data set we use is a bunch of Fairy tales. The output is shown below:

<code>data/tree.txt [('sloane', 0.089), ('kevin', 0.076), ('marina', 0.038), ('lake', 0.033), ('car', 0.02)]
data/rid.txt [('riddler', 0.198), ('daisy', 0.132), ('alien', 0.074), ('tequilla', 0.066), ('knowen', 0.066)]
data/jackbstl.txt [('jack', 0.124), ('ogre', 0.1), ('harp', 0.048), ('beanstalk', 0.045), ('hen', 0.035)]</code><hr><br><br>
  
  
## Word2vec<hr><br>
### Introduction<brr><br><br>
  
  * Word2vec (word to vector), as the name suggests, is a tool that converts words into vector form.

  * In all technicalities, Word2vec is a shallow two-layered neural network, that is used to produce the relative model of word embedding. It works by collecting a large number of vocabulary based datasets as input and outputs a vector space, such that each word in the dictionary maps itself to a unique vector. This allows us to represent the relationship between words.

  * Broadly, there are two models CBOW(Continuous Bag of Words) and Skip-Gram. Both neural network architectures essentially help the network learn how to represent a word. This is unsupervised machine learning, and labels are needed to train the model. Either of these two models, can create the labels for the given input and prepare the neural network to train the model and perform the desired task.
<img src='https://miro.medium.com/max/875/0*dyZ7Syt3DMbN7nF9' >
The difference in architecture between CBOW & Skip-gram models<br>
<hr>    
## CBOW<hr><br><br>

* CBOW(continuous bag-of-words) is a model suitable for work with smaller databases, and it does not need much RAM requirement either. In CBOW, we predict a word given its context. The entire vocabulary is used to create a “Bag of Words” model before proceeding to the next task.

    * The bag of words model is a simple representation of words that disregards grammar.

    Here is an example for better understanding.
        * John likes to watch movies. Mary likes movies too.
        * Mary also likes to watch football games.
        * A list of words is created by breaking the two sentences.
 
<img src = 'https://miro.medium.com/max/875/0*ozww2AL1RtblZsBA'>
* We then label each word with the number of occurrences. This is called a bag of words. In computers, this is usually captured as a JSON file.
<img src = 'https://miro.medium.com/max/875/0*zAHNwT93S17bL61w'>
* The sentences are then combined to get the overall frequency of each unique word.
<img src = 'https://miro.medium.com/max/875/0*k5o9wPgg9kl7MpCO'>
* Now that the bag of words model is ready, we can use CBOW to predict the probability of a word given these groups of words.
<br>
<img src = 'https://miro.medium.com/max/485/1*29PqUsL0nBcdxhu12JOiig.png'>    
CBOW model to predict probabilities of words<br><br>
    
* Each word and its frequency are passed as a unique vector into the input layer of the neural network. Say, if we have X words, the input layer takes in X[1XV] vectors and gives out 1[1XV] in the output layer.

* The input-hidden layer matrix sizes up to [VXN] and the output-hidden layer matrix sizes to [NXV]. In this case, N is the number of dimensions. The layers have no activation function between them.

* To calculate the output, the hidden input layers’ weights are multiplied by the hidden output layers weights. The error between the output and targets is calculated, and the weights are constantly readjusted through backpropagation. The only non-linearity is the softmax calculations in the output layer to generate probabilities for the word vectors.

* Overall, after calculations and readjustments, the weight between the hidden-output layer is taken as the word vector representation. As we can see, this architecture allows the model to predict the current word relying on influence from surrounding words.

    
<hr>## Skip Gram<hr><br><br>
    
* Suppose we have 10000 unique words, and we represent an input word like “ape” as a one-hot vector(A categorical word/variable can be better understood by an ML algorithm when it is one hot encoded to 0’s and 1's). This vector will have 10000 components; each component contains one vocabulary. And we place ‘1" in one position, which represents the word “ape” and place 0 in the rest of the positions.

    * The output of the network is a single vector with 10000 components as well. The probability that a randomly selected nearby word is that vocabulary word. We don’t need to consider the hidden layer neurons since none of them are active. However, the output neurons use softmax.

    * When we evaluate the trained network on an input word, the output is a probability distribution instead of a one-hot vector.

    
<br><hr>### The Hidden Layer<br><br>
    
* For example, we use a word vector with 300 features as the input. So the hidden layer should be a 300*10000 matrix, which means there are 10000 rows for words in vocabulary and 300 columns for hidden neurons.

    * The one-hot vector we use as input is used to pick out the corresponding row in the matrix. This means that the hidden layers are only used as a lookup table. And the output of the hidden layer is just the word vector for the input word.

<br><hr>### The Output Layer<br><br>
    
* As we pick up the word vector for “ape” in the hidden layers, it will be sent to the output layer. The output layer is a softmax regression classifier.
* So what is the softmax regression classifier? Softmax regression is a generalization of logistic regression that we can use for multi-class classification.
<img src = 'https://miro.medium.com/max/875/0*DzjiU2If05EqOIaV'>
Understanding softmax regression

    In softmax regression, all of the input components are classified in different classes. And the output is a one-hot vector.
<img src = 'https://miro.medium.com/max/875/0*dBgPz2W6YsfV6AJP'>
    
Implementing softmax for a word with several features

    * After the output vector multiple the weight vector from the hidden layer, it then applies the function exp(x) to the result. Finally, to get the outputs to sum up to 1, we need to divide the result by the sum of the result from all 10000 output nodes.

    * The words with a similar meaning are brought closer together in the vector space. The skip-gram model uses a word to predict the words surrounding it, and it relies on words with a closer context to function efficiently.
<hr><br><br>
## Differences between CBOW and Skip-Gram<br><br>
    
* Although the two models show mirror symmetry, they vary in terms of architecture and performance.

    1 - The Skip-Gram model predicts words surrounding a certain word by relying on the contextual similarity of words. On the other hand, the CBOW model uses the Bag of words approach and predicts a word using the words that surround it.

    2 - The Skip-Gram model is more accurate when it comes to infrequent words. It suits larger databases and requires more RAM to function. The CBOW model is faster, does not guarantee the handling of infrequent words, requires less RAM, and suits smaller databases.

    The choice of model depends on the user’s task.

### Training Tricks
To speed up the training of word2vec model, there are two ways you could try:

    
### Hierarchical Softmax
Now, the biggest problem is that we have a large amount of calculation from the hidden layer to the output softmax layer, because all words for softmax probability must be calculated, before finding the highest probability value. This model is shown below. And V represents the size of the glossary.
<img src = 'https://miro.medium.com/max/633/0*4yA5fjBeN9CWKlgf'>
Hierarchical Softmax
<br><br>    
### The first improvement:
For the mapping from the input layer to the hidden layer, a simple method of summing and averaging all input word vectors is used instead of a linear transformation of the neural network and an activation function. For instance, the input is three 4-dimensional word vectors:
        ( 1, 2, 3, 4 )
        ( 9, 6, 11, 8 )
        ( 5, 10, 7, 12 )
        Then the word vector after our word2vec mapping is
        ( 5, 6, 7, 8 )

### The second improvement:

    * It improves the number of calculations from the hidden layer to the output softmax layer. To avoid calculating all words for the softmax probability, word2vec uses the Huffman tree to replace mapping from the hidden layer to the output softmax layer.

    * Since we have converted all the probability calculations from the output softmax layer into a binary Huffman tree, our softmax probability calculation only needs to be performed along with the tree structure. As shown below, we can walk along the Huffman tree from the root node to the words of our leaf nodes W2.
<img src = 'https://miro.medium.com/max/485/0*vz5dT8ind6UQWPUq'>
Binary Huffman Tree for probability calculations

    All the internal nodes in our Huffman tree are similar to the neurons in the hidden layer of the neural network, where the word vector of the root node corresponds to our projected word vector, and all leaves nodes are similar to the neurons in the softmax output layer of the neural network. The number of leaf nodes is the size of the vocabulary. In the Huffman tree, the softmax mapping from the hidden layer to the output layer is step by step along the Huffman tree. Therefore, this softmax is named “Hierarchical Softmax.”

### Negative Sampling
* As mentioned in the previous section, millions of weights and tens of millions of training data mean that it is difficult to train this network.

    * So what we can do is to modify the optimization objective function. This strategy is called “Negative Sampling” so that each training sample only updates a small part of the weights in the model. It could reduce the number of train calculations but also improve the quality of the final word vector.
* We need to sample common words first.
<img src = 'https://miro.medium.com/max/875/0*iXFGauo-CL1FcpQh'>
Converting Source Text to Training Samples

    For common words like “the,” there will be two issues:
1 - For example, (fox, the) doesn’t convey our information about the fox. ‘The’ appears too much;
2 - We have too many (‘the’, …) samples, more than we need;

    * So word2vec uses a down-sampling strategy. For each word we encounter in the training sample, we have a probability of deleting it. This probability is called the “sampling rate” which is related to the frequency of words.
What is the sampling rate?

    * We use z(wi) to represent a word, and z(wi) represents the probability (frequency) that it appears in the thesaurus. For example, peanut appears 1,000 times in the 1bilion’s corpus, then z(peanut)=1E-6. Then there is a parameter called ‘sample’ that controls the degree of downsampling, which is generally set to 0.001. The smaller the value, the easier it is to throw away some words.
<img src = 'https://miro.medium.com/max/495/0*dUKnj5IMXwf4y69H'>
Equation of probability to retain a word

    * It can be seen that if z(wi)<=0.0026 P(wi)=1, then we will not throw these words away. If the occurrence frequency is very high, z(wi)==1(equivalent to nearly every training sample has the word), P(wi)=0.033, you can see that there is still a very low probability that we retain this word.
And then we do negative sampling.

                                          * Training a neural network means inputting a training sample to adjust the weight so that it predicts this training sample more accurately. In other words, each training sample will affect all weights in the network. As we discussed before, the size of our dictionary means that we have a lot of weight, all of which need to be adjusted slightly. Negative sampling solves this problem, and every time we change a small part of the weight, not all.

                                          * If the vocabulary size is 10000, when the input sample (“fox”, “quick”) is input to the neural network, “fox” is one-hot encoded, and in the output layer we expect the neuron node corresponding to the “quick” word to be output 1, the remaining 9999 should output 0. Here, the words corresponding to the 9999 neuron nodes that we expect to be 0 are negative words. The idea of ​​negative sampling is also very straightforward. A small number of negative words will be randomly selected, such as 10 negative words. Then update the corresponding weight parameters.

                                          * Assume that the original model requires 300 × 10,000 each time (in fact, there is no reduction in the number, but during the operation, the number of loads needs to be reduced.) Now only 300 × (1 + 10) is reduced a lot.

                                         Selecting negative samples:
* Here comes the question, how to choose 10 negative samples? The negative samples are also selected based on their probability of occurrence, and this probability is related to their frequency of occurrence. Words that appear more often are more likely to be selected as negative samples.
* This probability is expressed by a formula, and each word is given a weight-related to its frequency. The probability formula is:
<img src = 'https://miro.medium.com/max/325/0*U0kmPLfkQvz2fDfX'>
Equation of probability for weight-related frequencies
<br><br><br>
    
## Code tutorial
You can train your word2vec by genism easily. In the following example, we use a bunch of fairy tales as the data set. This code use word2vec model to find out familiar word for a specific word.
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
data_file="merged_clean.txt"
with open (data_file, 'r') as f:
    for i,line in enumerate(f):
        print(line)
        break
def read_input(input_file):
logging.info("reading file {0}...this may take a while".format(input_file))
with open(input_file, 'rb') as f:
        for i, line in enumerate(f):
if (i % 10000 == 0):
                logging.info("read {0} reviews".format(i))
            # do some pre-processing and return a list of words for each review text
            yield gensim.utils.simple_preprocess(line)
# read the tokenized reviews into a list
# each review item becomes a serries of words
# so this becomes a list of lists
documents = list (read_input(data_file))
logging.info ("Done reading data file")
model = gensim.models.Word2Vec (documents, size=150, window=10, min_count=2, workers=10)
model.train(documents,total_examples=len(documents),epochs=10)
w1 = "clean"
print(model.wv.most_similar (positive=w1))
The output is:
[('dry', 0.6215552091598511), ('neat', 0.5800684094429016), ('leather', 0.5562355518341064), ('cotton', 0.5484817624092102), ('heated', 0.5442177653312683), ('woollen', 0.5423670411109924), ('linen', 0.5314322710037231), ('tidy', 0.526747465133667), ('fur', 0.5222363471984863), ('starched', 0.5206629037857056)]
These words are considered to have some connection with ‘clean’.
<br><br><hr>
## Summary<hr><br>
    
    Two commonly-used word embedding techniques (TF-IDF and Word2vec). Each method includes an introduction, diagram, implementation, and elaboration with code on a real-world dataset. Further, we have also included Training Tricks (improvement methods) for the model. Word Embedding is the foundation of any major text analysis task, and we hope to have done justice to this topic by covering it in-depth.


