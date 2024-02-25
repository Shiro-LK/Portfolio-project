# Portfolio-project

This repository will present few projects i have made in the past.


- [Leaf Disease Detection Cassava](https://github.com/Shiro-LK/Portfolio-project/tree/main/Leaf_Disease_Detection_Cassava): This project was a kaggle Competition where the goal was to detect one of the 4 diseases categories of the leaf (or if it is healty). A gradio GUI can be found in this repo to infer on some samples.

- [Birdcalls Detection](https://github.com/Shiro-LK/Portfolio-project/tree/main/BirdsCall_Detection): This is a simplified version used during the BirdClef 2021 competition whose goal was to detect species of birds in the wild from audio clip. A gradio GUI can be found in this repo to infer on some samples wheter the audio contains a bird or not, or if it is part of one of the 575 species (pretrained model, not finetune one)

- [SoundEventDetection](https://github.com/Shiro-LK/SoundEventDetection): This custom package allows to create an architecture which can be used for sound event detection. I created it based on the different BirdClef competition which happens in the past. The idea is to transform a 1D signal and get an mel spectrogram using torchlibrosa. Then we can use a CNN backbone in order to extract relevant features maps. Finally, we can use a SED block based on attention and get our final output.

- [Movie Genre Prediction](https://github.com/Shiro-LK/CompetitionsML/tree/main/huggingface_competition/movie-genre-prediction) This project was a Huggingface Competition whose goal was to design a predictive model which can classifies movies into their respective genres. The training script has been shared for reproduction.

- [tflibrosa](https://github.com/Shiro-LK/tflibrosa) re-implementation of the torchlibrosa package, in tensorflow.

- [DOLG](https://github.com/Shiro-LK/python-DOLG) re-implementation of DOLG architecture in tensorflow. This architecture has been used for landmark recognition, as it seems to perform well for local and global features.

- [folcanet](https://github.com/Shiro-LK/focalnet-tf) re-implementation of the focalnet architecture in tensorflow.