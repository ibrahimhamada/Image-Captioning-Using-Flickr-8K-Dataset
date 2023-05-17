# Image-Captioning-Using-Flickr-8K-Dataset
Project of the Neural Networks and Deep Learning Course Offered in Spring 2022 @ Zewail City


In this project, we are trying to generate meaningful descriptions for images. 
It is a very important task in the text generation domain, in which the goal is generating meaningful sentences in the form of human-written text. Image captioning has plenty of applications such that using image captions to create an application to help people who have low or no eyesight.



## Dataset:

The [Flickr 8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) will be used in training and testing the proposed model. The Flickr 8k dataset has been widely used in the field of sentence-based image description. 
The dataset consists of 8000 images that capture people engaged in everyday events and activities, where each image is annotated using 5 different sentences provided by human annotators. Accordingly, the dataset contains around 40,000 captions for 8,000 images. The pictures were hand-picked to represent a diversity of events and circumstances from six different Flickr groups, and they usually don't feature any famous individuals or places. We will be using 6000 images for training, 1000 images for validation and another 1000 images for testing.

Here are some dataset samples:

![image](https://github.com/ibrahimhamada/Image-Captioning-Using-Flickr-8K-Dataset/assets/58476343/c448fb96-89ac-4802-ab01-0257915fd1b8)
![image](https://github.com/ibrahimhamada/Image-Captioning-Using-Flickr-8K-Dataset/assets/58476343/05c4eb80-1b9f-4060-8f67-3f667445e95e)


## Data Preprocessing:

We did the following Data Preprocessing :
  1. Lowercase all the words
  2. Remove Punctuations
  3. Encapsulates the captions start and end sequences
  4. Vectorize words with tokenze
  5. Prepare a dictionary that has key = imageID and value= List of captions which are vectorized
     - We encapsulated the captions so the model can know when the caption ends and when it starts.
     - We also created a vocabulary which size was 8917 and after doing the necessary data cleaning it was 8357.
     - In figure 6, we can see some statistics about the data set before data cleaning, as for example we can see that a is the most frequently appearing word and we are going to remove it since it does not add much information.

![image](https://github.com/ibrahimhamada/Image-Captioning-Using-Flickr-8K-Dataset/assets/58476343/0cd0740e-eadb-47b5-911b-396ac31475f8)

- In figure 7, we can see some images alongside their captions after all the data cleaning that we did.

![image](https://github.com/ibrahimhamada/Image-Captioning-Using-Flickr-8K-Dataset/assets/58476343/073aca2d-e688-4efb-91d9-c66161fd0112)



## Architecture:

Our architecture receives two inputs which are image features and input words. Then it tries to predict the next word. Consequently, our architecture can be divided into three main parts, the first part is the image model, the second one is the language model, and the last one is a concatenation of both.

The image model is composed of 3 layers as shown in figure 8 below. We have used a dense layer as a way to learn what are the important features to be used. Then we followed it with a Repeat vector layer to repeat the image features and pass it to all the steps of the GRUs. Finally, we used a dropout layer as a regularizer with a rate of 0.4. The image model takes image features with a size = of 2048. We have tried to extract the features with two models one with Resnet 50 and the other with Xception.

![image](https://github.com/ibrahimhamada/Image-Captioning-Using-Flickr-8K-Dataset/assets/58476343/03588518-e0d4-4a8c-adba-362578551a90)

The language model is composed of 2 layers as shown in figure 9 below. We have used an embedding layer to learn the text representation of our specified dataset. Then before going to the next layer we used a lambda function to remove masks from the embedding layers as this would cause an issue when we concatenate with the image model. Finally, we followed it with a dropout layer to act as a regular. Instead of passing the words to the language model we tried to initialize the values of the word with three different models and compared the results between them. We first used an embedding layer with dim=50. Then we tried using a glove model with dim= 50 and finally, we used another glove model with dim=300.

![image](https://github.com/ibrahimhamada/Image-Captioning-Using-Flickr-8K-Dataset/assets/58476343/329d0ebe-2872-495b-a636-109849106e56)

The third part of our model is a concatenation between the two models. After concatenation, we had four layers which are a GRU followed by a dropout layer then another GRU layer, and finally a dense layer. Although most of the papers we have seen were using LSTMs, we thought that it would be better to use GRUs as we would have a smaller number of learning parameters. Consequently, we would have fewer calculations and a smaller model size with almost the same approximation results.

![image](https://github.com/ibrahimhamada/Image-Captioning-Using-Flickr-8K-Dataset/assets/58476343/91a22fb0-79dd-4692-925b-5213114bba7a)

