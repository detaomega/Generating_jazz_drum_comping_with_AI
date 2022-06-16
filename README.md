# Generating_jazz_drum_comping_with_AI
2022 AI spring projecj
## Introduction
Our topic is generating jazz drum comping by neural network. Comping ideas are designed to accompany and complement both the swinging ride cymbal and the entire band." According to one of our members, who is a drummer with experiences of jazz drumming, the comping ideas and rhythms applied are usually based on the melody that is currently playing. As a result, we want to see if we can train an AI to capture the relationship between the melody and the comping, more specifically, the comping of snare and bass drums.
## Prerequisite
The requirements are listed in requirement.txt
You can set up your environment by using the command
```
pip install -r requirements.txt
```
## Dataset
We choose the midi file type as our way to represent music, because compared
to .mp3 or .wav files, we can easily get the information on
timestamps, duration and pitches of individual notes using
this storage format. In our program, we use the pretty midi
python model to process the midi file.
## Baseline
Convolutional Neural Network(CNN)
## Main approach
Recurrent Neural Network(RNN)
## To training the model
Run train_model.py 
```
python prototype.py
```
there are two type that can choose to train
* CNN
* RNN
## Result
![image](https://user-images.githubusercontent.com/79638758/174083949-9e00a4c9-78ba-44e0-a39b-060d322866e3.png)

![image](https://user-images.githubusercontent.com/79638758/174084054-76e990ff-b9a9-4dee-8d0e-27cb6fbdf96f.png)

* generate by CNN
1. [Song1](https://youtu.be/0DcI3PNIeAU)
2. [Song2](https://youtu.be/VVzAsTQRY40)
3. [Song3](https://youtu.be/9UXhb1d-4k8)
4. [Song4](https://youtu.be/D4abqDvi_l4)
5. [Song5](https://youtu.be/J-QtVln2I8U)
* generate by RNN
1. [Song1](https://youtu.be/8zYzmaIrTqg)
2. [Song2](https://youtu.be/JdM0MqNYrck)
3. [Song3](https://youtu.be/QsbxA_S1VdY)
4. [Song4](https://youtu.be/-_DVaZJHcNA)
5. [Song5](https://youtu.be/2Cay9RrQwsw)

* For comparison, these are the original songs
  [Link](https://github.com/detaomega/Generating_jazz_drum_comping_with_AI/tree/main/sample/NoDrum%20)
