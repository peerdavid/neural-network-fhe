#
# From https://docs.zama.ai/concrete-numpy/stable/user/advanced_examples/FullyConnectedNeuralNetwork.html#Compile-the-model
#

import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from concrete.torch.compile import compile_torch_model


class FCIris(torch.nn.Module):
    """Neural network for Iris classification

    We define a fully connected network with three (3) fully connected (fc) layers that
    perform feature extraction and one (fc) layer to produce the final classification.
    We will use 3 neurons on all layers to ensure that the FHE accumulators
    do not overflow (we are currently only allowed a maximum of 7 bits-width).
    More information on this is available at
    https://docs.zama.ai/concrete-numpy/main/user/howto/reduce_needed_precision.html#limitations-for-fhe-friendly-neural-network.

    Due to accumulator limits, we have to design a network with only a few neurons on each layer.
    This is in contrast to a traditional approach where the number of neurons increases after
    each layer or block.
    """

    def __init__(self, input_size):
        super().__init__()

        # The first layer processes the input data, in our case 4 dimensional vectors
        self.linear1 = nn.Linear(input_size, 3)
        self.sigmoid1 = nn.Sigmoid()
        # Next, we add a one intermediate layer
        self.linear2 = nn.Linear(3, 3)
        self.sigmoid2 = nn.Sigmoid()
        # Finally, we add the decision layer for 3 output classes encoded as one-hot vectors
        self.decision = nn.Linear(3, 3)

    def forward(self, x):

        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.linear2(x)
        x = self.sigmoid2(x)
        x = self.decision(x)

        return x


def train(n_iters, X_train, y_train, batch_size, model, criterion, optimizer):
    for i in range(n_iters):
        # Get a random batch of training data
        idx = torch.randperm(X_train.size()[0])
        X_batch = X_train[idx][:batch_size]
        y_batch = y_train[idx][:batch_size]

        # Forward pass
        y_pred = model(X_batch)

        # Compute loss
        loss = criterion(y_pred, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        if i % 1000 == 0:
            # Print epoch number, loss and accuracy
            accuracy = torch.sum(torch.argmax(y_pred, dim=1) == y_batch).item() / y_batch.size()[0]
            print(f"Iterations: {i:02} | Loss: {loss.item():.4f} | Accuracy: {100*accuracy:.2f}%")
            if accuracy == 1:
                break


#
# M A I N
#
def main():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Convert to tensors
    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    # Initialize our model
    model = FCIris(X.shape[1])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    n_iters = 50001
    batch_size = 16

    #
    # TRAIN
    #
    print("Training a FHE friendly quantized network.")

    # We need a retry logic here: In some cases, the network accumulators
    # can overflow and in this case we simply retrain the model.
    for _ in range(10):
        try:
            train(n_iters, X_train, y_train, batch_size, model, criterion, optimizer)
            print("Compiling the model to FHE.")
            quantized_compiled_module = compile_torch_model(
                model,
                X_train,
                n_bits=3,
            )
            print("The network is trained and FHE friendly.")
            break
        except RuntimeError as e:
            if str(e).startswith("max_bit_width of some nodes is too high"):
                print("The network is not fully FHE friendly, retrain.")
                train()
            else:
                raise e

    #
    # Predict
    #
    y_pred = model(X_test)

    #
    # Predict quantized
    #
    X_train_numpy = X_train.numpy()
    X_test_numpy = X_test.numpy()
    y_train_numpy = y_train.numpy()
    y_test_numpy = y_test.numpy()
    q_X_test_numpy = quantized_compiled_module.quantize_input(X_test_numpy)
    quant_model_predictions = quantized_compiled_module(q_X_test_numpy)

    #
    # Predict in FHE
    #
    fhe_x_test = quantized_compiled_module.quantize_input(X_test_numpy)
    homomorphic_quant_predictions = []
    for x_q in tqdm(fhe_x_test):
        homomorphic_quant_predictions.append(
            quantized_compiled_module.forward_fhe.run(np.array([x_q]).astype(np.uint8))
        )
    homomorphic_predictions = quantized_compiled_module.dequantize_output(
        np.array(homomorphic_quant_predictions, dtype=np.float32).reshape(quant_model_predictions.shape)
    )

    acc_0 = 100 * (y_pred.argmax(1) == y_test).float().mean()
    acc_1 = 100 * (quant_model_predictions.argmax(1) == y_test_numpy).mean()
    acc_2 = 100 * (homomorphic_predictions.argmax(1) == y_test_numpy).mean()

    print(f"Test Accuracy: {acc_0:.2f}%")
    print(f"Test Accuracy Quantized Inference: {acc_1:.2f}%")
    print(f"Test Accuracy Homomorphic Inference: {acc_2:.2f}%")



if __name__ == '__main__':
    main()