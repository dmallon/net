package neuralNet;

// Training strategy for traditional Backpropagation
public class BPTrainingStrategy implements ITrainingStrategy {

	// MLPNN Training method that uses BP algorithm
	@Override
	public void train(MLPNet net, double[][] trainSet, int numSamples, int epochs, char[] classes) {
		double expected;
		double[] firstIn = new double[net.numInputs];
		double[] inputs = new double[net.numNodes];
		double[] outputs = new double[net.numNodes];
		double[] finalOut = new double[net.numOutputs];
		double sum;
		
		int a = 0;
		// Loop for all epochs
		while (a < epochs){
			// Loop through all samples
			for (int i = 0; i < numSamples; i++){
				
				// Get the input from the current training vector
				for (int j = 0; j < net.numInputs; j++){
					firstIn[j] = trainSet[j][i];
				}
				
				// Check the expected output
				if(net.classifier == classes[i])
					expected = 1.0;
				else
					expected = 0.0;
				
				// Activate hidden layers
				for (int j = 0; j < net.numLayers; j++){
					for (int k = 0; k < net.numNodes; k++){
						if(j == 0)
							net.hidden[k][j].activate(firstIn);
						else
							net.hidden[k][j].activate(inputs);
						outputs[k] = net.hidden[k][j].getOutput();
					}						
					for (int k = 0; k < net.numNodes; k++){
						inputs[k] = outputs[k];
					}
				}
				
				// Activate Output layer (and calculate output delta)
				for (int j = 0; j < net.numOutputs; j++){
					net.output[j].activate(inputs);
					finalOut[j] = net.output[j].getOutput();
					net.output[j].setDelta(finalOut[j] - expected);
				}
				
				// Begin back-propagation of delta at Output nodes				
				for (int j = 0; j < net.numNodes; j++){
					sum = 0.0;
					for (int k = 0; k < net.numOutputs; k++){
						sum += net.output[k].getDelta() * net.output[k].getWeight()[j];
					}
					net.hidden[j][net.numLayers - 1].setDelta(net.hidden[j][net.numLayers - 1].getDerivative() * sum);
				}			
				// Backprop through hidden nodes			
				for (int j = net.numLayers - 2; j >= 0; j--){
					for(int k = 0; k < net.numNodes; k++){
						sum = 0.0;
						if(j > 0){
							for (int l = 0; l < net.numNodes; l++){
								sum += net.hidden[l][j + 1].getDelta() * net.hidden[l][j + 1].getWeight()[k]; 
							}
						}
						else{
							for (int l = 0; l < net.numInputs; l++){
								sum += net.hidden[l][j + 1].getDelta() * net.hidden[l][j + 1].getWeight()[k]; 
							}
						}
						net.hidden[k][j].setDelta(sum);
					}
				}
				
				// Begin forward propagation of weights
				for (int j = 0; j < net.numInputs; j++){	
					firstIn[j] = trainSet[j][i];
				}
				
				Neuron n;
				double wPrime;
				// Set the weights for the hidden layers
				for (int j = 0; j < net.numLayers; j++){
					if(j == 0){
						for (int k = 0; k < net.numInputs; k++){
							n = net.hidden[k][j];						
							for (int l = 0; l < net.numInputs; l++){
								wPrime = n.getWeight()[l] - net.rate * n.getDelta() * n.getDerivative() * firstIn[l];
								n.setWeight(l, wPrime);
							}
						}
						for (int k = 0; k < net.numInputs; k++){
							inputs[k] = net.hidden[k][j].getOutput();
						}
					}
					else{
						for (int k = 0; k < net.numNodes; k++){
							n = net.hidden[k][j];						
							for (int l = 0; l < net.numNodes; l++){
								wPrime = n.getWeight()[l] - net.rate * n.getDelta() * n.getDerivative() * inputs[l];
								n.setWeight(l, wPrime);
							}
						}
						for (int k = 0; k < net.numNodes; k++){
							inputs[k] = net.hidden[k][j].getOutput();
						}
					}
				}
				// Set the weights for the output layer
				for (int j = 0; j < net.numOutputs; j++){
					n = net.output[j];
					for (int k = 0; k < net.numNodes; k++){
						wPrime = n.getWeight()[k] - net.rate * n.getDelta() * n.getDerivative() * inputs[k];
						n.setWeight(k, wPrime);
					}
				}
				// Increment epoch counter
				a++;
				// Exit loop if number of epochs has been reached
				if(a == epochs)
					break;
			}
		}
		
	}

}
