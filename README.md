# Weather-prediction-

The aim of this project is to train and test sequence to sequence networks of given parameters and then generating weather status.   


I have used tensorflow to train a sequence-to-sequence model using the encoder-decoder archi-
tecture. The overall structure of the network is as follows:
---=>
INEMBED: A feedforward layer of size |V s | × inembsize, where V s is the source vocabulary (i.e. set of words) and inembsize is               the output size of the layer. Use inembsize = 256

ENCODER- bidirectional LSTM layer with encsize outputs in either direction. encsize = 512

DECODER: LSTM layer with decsize outputs. decsize = 512

ATTENTION: Incorporate an attention mechanism over the Basic Encoder

SOFTMAX: softmax layer for classification: decsize inputs and V t outputs,where V t is the target language vocabulary (i.e. set of words in the summary).

OUTEMBED: A feedforward layer of size |V t | × outembsize, where embsize is the output size of the layer. Use outembsize = 256. This layer is used for obtaining the embedding of the output character and feeding it to the input ofthe decoder for the next time step.
----=>

I have train the network using Adam on the entire training dataset. Use the valid set for validation.
Dropout is applied on the output of the encoder and the decoder when training the network.

Used early stopping on the validation set with a patience of 5 epochs.
performance is maeusered by BLEU-4 score.
