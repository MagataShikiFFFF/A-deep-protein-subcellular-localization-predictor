# A deep protein Subcellular-localization predictor
This is a subcellular localization predictor with an enhanced feature extractor by transfer learning.
It is consisting of a linear classifier and a deep feature extractor of convolution neural network (CNN). The deep CNN feature extractor is first shared and pre-trained in a deep GO annotation predictor, and then is transferred to the subcellular localization predictor with fine-tuning using protein localization samples. 

 It is built based on the framework chainer.
The datasets are in folder n-gram.
