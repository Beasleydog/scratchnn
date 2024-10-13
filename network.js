const {
  dot,
  add,
  multiply,
  subtract,
  transpose,
  matrix,
  dotMultiply,
  exp,
} = require("mathjs");

class Network {
  constructor(layerSizes) {
    this.num_layers = layerSizes.length;
    this.layers = layerSizes;

    this.weights = [];
    this.biases = [];

    this.errors = [];
    this.weightedInputs = [];
    this.activations = [];

    layerSizes.forEach((size, layerNumber) => {
      if (layerNumber === 0) return; // Skip input layer

      const previousLayerSize = layerSizes[layerNumber - 1];
      const currentLayerSize = size;

      // Initialize weights
      const weightLayer = Array.from({ length: currentLayerSize }, () =>
        Array.from({ length: previousLayerSize }, () => Math.random())
      );
      this.weights.push(weightLayer);

      // Initialize biases
      const biasLayer = Array.from({ length: currentLayerSize }, () => [
        Math.random(),
      ]);
      this.biases.push(biasLayer);
    });
  }

  // Activation function
  activate = (z) => {
    return z.map(sigmoid);
  };

  // Derivative of activation function
  activatePrime = (z) => {
    //Z is an array of values
    return z.map(sigmoidPrime);
  };

  loss(predictions, targets) {
    // Currently using MSE, could be changed to other loss functions
    return (
      predictions.reduce((sum, prediction, i) => {
        return sum + pow(prediction - targets[i], 2);
      }, 0) / predictions.length
    );
  }

  feedforward(values) {
    this.weightedInputs = [];
    this.activations = [];

    // Loop through each layer, passing the value along
    this.weights.forEach((allWeightsForAllNodesInLayer, weightLayerIndex) => {
      const weightMatrix = matrix(allWeightsForAllNodesInLayer);
      const valueVector = matrix(values);
      const biasVector = matrix(this.biases[weightLayerIndex]);

      // Perform matrix multiplication and add biases
      const justWeightedInput = multiply(weightMatrix, valueVector);
      const z = add(justWeightedInput, biasVector);
      this.weightedInputs.push(z);
      // Apply activation function element-wise
      values = this.activate(z);
      this.activations.push(values);

      console.log("adding activations from layer", weightLayerIndex);
    });
    return values;
  }

  costDerivative(outputActivations, targetActivations) {
    //MSE Equation: sum((y_i - y_hat_i)^2) / n
    //MSE Derivative: 2(y_hat - y)
    return multiply(2, subtract(outputActivations, targetActivations));
  }

  //Training data is 2xN matrix, where the first vector is the input and the second is the target output
  stochasticGradientDescent(trainingData, epochs, batchSize, learningRate) {
    const shuffledData = trainingData.sort(() => Math.random() - 0.5);
    const miniBatches = [];
    for (let i = 0; i < shuffledData.length; i += batchSize) {
      miniBatches.push(shuffledData.slice(i, i + batchSize));
    }
    for (let epoch = 0; epoch < epochs; epoch++) {
      console.log(`Epoch ${epoch + 1}/${epochs}`);
      miniBatches.forEach((miniBatch, index) => {
        console.log(`Processing miniBatch ${index + 1}/${miniBatches.length}`);
        this.handleBatch(miniBatch, learningRate);
      });
    }
    console.log("Training complete");
  }

  handleBatch(miniBatch, learningRate) {
    let weightGradients = this.weights.map((layer) =>
      layer.map((row) => row.map(() => 0))
    );
    let biasGradients = this.biases.map((layer) =>
      layer.map((bias) => bias * 0)
    );

    miniBatch.forEach(([input, target]) => {
      const { deltaWeights, deltaBiases } = this.backpropagate(input, target);
      weightGradients = weightGradients.map((weight, weightIndex) =>
        add(weight, deltaWeights[weightIndex])
      );
      biasGradients = biasGradients.map((bias, biasIndex) =>
        add(bias, deltaBiases[biasIndex])
      );
    });

    const mappedWeights = weightGradients.map((x) =>
      multiply(x, learningRate / miniBatch.length)
    );
    const mappedBiases = biasGradients.map((x) =>
      multiply(x, learningRate / miniBatch.length)
    );
    this.weights = this.weights.map((layer, i) =>
      subtract(layer, mappedWeights[i])
    );
    this.biases = this.biases.map((layer, i) =>
      subtract(layer, mappedBiases[i])
    );
  }

  backpropagate(input, target) {
    console.log("input", input);
    const activations = this.feedforward(input);

    //Calculate the error of the output layer
    //Note that we subtract two because weightedInputs will always have one less element than # layers
    const outputError = dotMultiply(
      subtract(matrix(activations), matrix(target)),
      matrix(this.activatePrime(this.weightedInputs[this.num_layers - 1 - 1]))
    );

    //Calculate the error of the hidden layers
    //Note that we subtract one because there is one less errors than there are layers
    let errors = new Array(this.num_layers - 1);
    errors[errors.length - 1] = outputError;

    //Start from the second to last layer and go backwards
    for (let i = this.num_layers - 1 - 1; i >= 1; i--) {
      //Note that to calculate the error of the current layer we need values from the layer AFTER it.
      //This code references the after layer as the "next" layer

      //Note that it appears we should be using i+1 at first (because we are trying to get the next layers weighted inputs and weights)
      //However, since there is #layers-1 inputs in the model, we only need to grab the i'th value
      const nextWeightedInputs = this.weightedInputs[i];
      const nextWeights = this.weights[i];

      //Note that we just use i here.
      //You would expect i+1 because we're getting the next error however since there is one less error than there are layers we have to subtract one
      //This just leaves us with i
      const nextError = errors[i + 1 - 1];

      const currentLayerErrors = dotMultiply(
        multiply(transpose(nextWeights), nextError),
        matrix(this.activatePrime(nextWeightedInputs))
      );
      //Note that we subtract one because there is one less errors than there are layers
      errors[i - 1] = currentLayerErrors;
    }

    let deltaWeights = this.weights.map((layer) =>
      layer.map((row) => row.map(() => 0))
    );
    let deltaBiases = this.biases.map((layer) => layer.map((bias) => bias * 0));

    //Remember that we are in the POV of last layer looking backwards
    for (let layer = this.num_layers - 1; layer > 0; layer--) {
      //Note that we subtract one because there is one less errors than there are layers
      const currentError = errors[layer - 1];

      //We want the previous layer's activations so you would assume it would be layer-1 but because there is one less activation than there are layers, we have to do layer-1-1
      const previousActivations = this.activations[layer - 1 - 1] || input;

      console.log("current error", currentError);
      console.log("previous activations", previousActivations);
      console.log("transpose", transpose(matrix(previousActivations)));

      deltaWeights[layer - 1] = add(
        deltaWeights[layer - 1],
        multiply(currentError, transpose(matrix(previousActivations)))
      );
      deltaBiases[layer - 1] = add(deltaBiases[layer - 1], currentError);
    }

    return { deltaWeights, deltaBiases };
  }

  save(filename) {
    const fs = require("fs");
    const data = JSON.stringify({ weights: this.weights, biases: this.biases });
    fs.writeFileSync(filename, data);
    console.log(`Network parameters saved to ${filename}`);
  }

  load(filename) {
    const fs = require("fs");
    const data = fs.readFileSync(filename, "utf8");
    const { weights, biases } = JSON.parse(data);
    this.weights = weights;
    this.biases = biases;
    console.log(`Network parameters loaded from ${filename}`);
  }

  // Helper method for input validation
  validateArray(arr, arrName) {
    if (!Array.isArray(arr)) {
      throw new Error(`${arrName} is not an array.`);
    }
    arr.forEach((item, index) => {
      if (typeof item !== "number" || isNaN(item)) {
        throw new Error(`${arrName}[${index}] is not a valid number.`);
      }
    });
  }
}

// Activation Functions
function sigmoid(z) {
  return 1.0 / (1.0 + exp(-z));
}

function sigmoidPrime(z) {
  const s = sigmoid(z);
  return s * (1 - s);
}

module.exports = Network;
