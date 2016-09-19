#Deep Learning for NLP

A list of resources dedicated to deep learning for natural language processing tasks


##Word Vector
1. Bengio Y, Schwenk H, Senécal J S, et al. [A Neural Probabilistic Language Models](http://www.cs.columbia.edu/~blei/seminar/2016_discrete_data/readings/BengioDucharmeVincentJanvin2003.pdf)[J]. Journal of Machine Learning Research, 2003, 3(6):1137-1155.  
—— Introduction to a neural langauge model that learns a distributed representation for each word, along with the probability function for word sequence.

2. Morin F, Bengio Y. [Hierarchical probabilistic neural network language model](https://www.researchgate.net/publication/228348202_Hierarchical_probabilistic_neural_network_language_model)[J]. Aistats, 2005.  
—— A hierarchical neural network called hierasrchical softmax that provides exponential speed-up when used to compute conditional probabilities.

3. Mikolov T, Chen K, Corrado G, et al. [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)[J]. Computer Science, 2013.  
—— CBOW and Skip-gram are two new log-linear model architectures for learning distributed representations of words. They can be used for learning high-quality word vectors from huge data sets with billions of words, and with millions of words in the vocabulary.  
of words and a 

4. Gutmann M U, Hyv&#, Rinen A. [Noise-contrastive estimation of unnormalized statistical models, with applications to natural image statistics](http://jmlr.org/papers/volume13/gutmann12a/gutmann12a.pdf)[J]. Journal of Machine Learning Research, 2012, 13(1):307-361.  
—— Noise-Contrastive Estimation(NCE) is an is an objective function for estimation of both normalized and unnormalized models, a simplified verson of NEC called Negative Sampling Estimation(NSC) is applied on Word2Vec to speed up training. 

5. Mikolov T, Sutskever I, Chen K, et al. [Distributed Representations of Words and Phrases and their Compositionality](http://arxiv.org/pdf/1310.4546v1.pdf)[J]. Advances in Neural Information Processing Systems, 2013, 26:3111-3119.  
—— This paper discribes architecture of Google's word2vec, it is an extension of Skip-gram models with subsampling of frequent words and NSC as an alternation to the hierarchical softmax. 

6. Goldberg Y, Levy O. [word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method](http://de.arxiv.org/pdf/1402.3722v1)[J]. Eprint Arxiv, 2014.  
—— Detailed description of Negative Sampling Estimation

7. Pennington J, Socher R, Manning C. [Glove: Global Vectors for Word Representation](http://www.aclweb.org/website/anthology/D/D14/D14-1162.pdf)[C]// Conference on Empirical Methods in Natural Language Processing. 2014.  
—— Glove is a global logbilinear regression model that combines the advantages of the two major model families:global matrix factorization and local context window methods

8. Bojanowski P, Grave E, Joulin A, et al. [Enriching Word Vectors with Subword Information](http://arxiv.org/pdf/1607.04606v1.pdf)[J]. 2016.  
—— Langusge model for Facebook's FastText, it is an extension of Skip-gram model and propose a different scoring function that take into account of internal structure of words.
9. [word2vec中的数学原理详解](http://blog.csdn.net/itplus/article/details/37969519)   
—— A Chinese blog for word2vec

##Tools
1. [Gensim](http://radimrehurek.com/gensim/index.html) is a free Python library designed to automatically extract semantic topics from documents, as efficiently (computer-wise) and painlessly (human-wise) as possible.It can also be used to train word2vec models
2. [TensorFlow](https://www.tensorflow.org/) is an open source software library for machine intelligence,The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API.



