import torch
import torch.nn as nn
import utils

class Decoder(nn.Module):
    """
    A class representing a Decoder module for sequence prediction.

    This class implements a decoder with Long Short-Term Memory (LSTM) cells
    and a Koopman inverse layer to predict the next steps in a sequence.

    Args:
        data_steps_used_for_prediction (int): Number of steps used for prediction.
        hidden_size (int): Size of the hidden state of the LSTM.
        num_layers (int): Number of LSTM layers.

    Attributes:
        embed_size (int): Size of the input data for prediction.
        hidden_size (int): Size of the hidden state of the LSTM.
        num_layers (int): Number of LSTM layers.
        lstm (nn.LSTM): LSTM module for sequence decoding.
        koopman_inverse (nn.Linear): Linear transformation for Koopman inverse.

    Methods:
        forward(input, decoder_hidden, decoder_cell):
            Forward pass through the Decoder model.
    """

    def __init__(self,
                 data_steps_used_for_prediction : int,
                 hidden_size                    : int,
                 num_layers                     : int = 2,
                 ) -> None:
        """
        Initialize the Decoder instance.

        Args:
            data_steps_used_for_prediction (int): Number of steps used for prediction.
            hidden_size (int): Size of the hidden state of the LSTM.
            num_layers (int): Number of LSTM layers.
        """
        super(Decoder, self).__init__()
        self.embed_size = data_steps_used_for_prediction
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=True)
        self.koopman_inverse = nn.Linear(self.hidden_size, 1)

    def forward(self,
                input           :torch.Tensor,
                decoder_hidden  :torch.Tensor,
                decoder_cell    :torch.Tensor,
                )->tuple:
        """
        Perform a forward pass through the Decoder module.

        Args:
            input (torch.Tensor): Input data tensor.
            decoder_hidden (torch.Tensor): Hidden state of the decoder.
            decoder_cell (torch.Tensor): Cell state of the decoder.

        Returns:
            decoder_output (torch.Tensor): Output prediction tensor.
            (decoder_hidden, decoder_cell): Updated decoder hidden and cell states.
        """
        decoder_output, (decoder_hidden, decoder_cell) = self.lstm(input, (decoder_hidden, decoder_cell))

        decoder_output = self.koopman_inverse(decoder_output)

        return decoder_output, (decoder_hidden, decoder_cell)

    

class Decoder_CNN(nn.Module):
    def __init__(self,
                 ) -> None:
        super(Decoder_CNN, self).__init__()

        self.modell =   nn.Sequential(
                        nn.Conv1d(1,1,400),
                        nn.ReLU(),
                        nn.Conv1d(1,1,300),
                        nn.ReLU(),
                        # nn.Conv1d(1,1,102),
                        # nn.ReLU()
                        )

        self.ffl = nn.Sequential(nn.Linear(self._test(),1),
                                 nn.ReLU(),
        )
        # self.koopman_inverse = nn.Linear(self.hidden_size, 1)

    def _test(self,):
        return self.modell(torch.randn(1,1,utils.prediction_input_size*4)).shape[-1]


    def forward(self,
                input           :torch.Tensor,
                )->tuple:
        decoder_output = self.ffl(self.modell(input))
        return decoder_output