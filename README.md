# UDACITY (Computer Vision Nanodegree) - Image Captioning

 In this project, we build a neural network based on the following paper https://arxiv.org/pdf/1411.4555.pdf. 
 <br>Architecture of net consists of two parts, Encoder and Decoder. Encoder serves as a feature extractor and decoder serves for generating sequences. In this project, the decoder consisted of LSTM layers with dropout.
 <br>The net was trained on captions for MS COCO dataset.
 <br>For evaluation of neural network performance, we used perplexity. It measures how well the neural network predits the sample. A low perplexity means that NN is good at predicting captions for samples.
 
 <br>The result of training was automatically generated captios for sample images that were in many cases quite accurate...<br>
![alt text](https://github.com/koles289/udacity-Image-Captioning/blob/master/Good_example.png?raw=true)
 
 <br>but in some cases it was completly wrong.<br>
![alt text](https://github.com/koles289/udacity-Image-Captioning/blob/master/bad_example.png?raw=true)

<br>Obvioulsly, the neural network can not generate accurate caption for objects that it sees for the first time and the result of this behavior are many funny captions...
