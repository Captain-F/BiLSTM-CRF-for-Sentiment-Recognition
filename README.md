# BiLSTM-CRF-for-Sentiment-Recognition
Employing BiLSTM-CRF model to classify the sentiment of texts. 
The sentiments of texts are positvie, neutral and negative.

Here, I dont't describe the theory of BiLSTM-crf because there is too much resource on the Internet. So just go stright to the goal.

Goal: Classifying the sentiments of texts into positvie, neutral and negative.

Input:
word representation: Using the word2vec to represent the words. The dimension of vectors is 100. The length of sentence is 50. If the length of texts is beyond 50, we cut it off. Otherwise, we use zero padding strategy.
So the shape of sentence is (50, 100)

Label:
Unlike the accepted labels form of keras and pytorch, the dimension of labels in Bilstm-crf is 3 and the labels are in a form of [[[0]],[[1]],...], which confuses me for a litle time. 
Example:
In keras(Bilstm), we get 2 texts, whose labels are postive and negtive. (we use 0 as the label of positive and 1 as the label of negtive)
Their form: [[1,0,0],[0,1,0]]
In Bilstm-crf(keras)
Their form: [[[0]],[[1]]]
So the shape of label is (1, 1)

Output:
Classification report, containing P, R, F metrics.

Language:
Python >= 3.6

Denpendent libraries:
scikit-learn >= 0.20.0
keras >= 2.2.4
keras-contrib >= 2.0.8
