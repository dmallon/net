package neuralNet;

public class BPTrainingStrategy implements ITrainingStrategy {

	@Override
	public void train(double[][] trainSet, int numSamples, int epochs, char[] classes, int numInputs, int numLayers, int numNodes, int numOutputs, Neuron[][] hidden, Neuron[] output, char classifier, double rate) {
		double expected;
		double[] firstIn = new double[numInputs];
		double[] inputs = new double[numNodes];
		double[] outputs = new double[numNodes];
		double[] finalOut = new double[numOutputs];
		double sum;
		
		int a = 0;
		
		while (a < epochs){
			for (int i = 0; i < numSamples; i++){
				rate = (rate / 1.000001);
				
				for (int j = 0; j < numInputs; j++){
					firstIn[j] = trainSet[j][i];
				}
				
				if(classifier == classes[i])
					expected = 1.0;
				else
					expected = 0.0;
				
				// Hidden layers
				for (int j = 0; j < numLayers; j++){
					for (int k = 0; k < numNodes; k++){
						if(j == 0)
							hidden[k][j].activate(firstIn);
						else
							hidden[k][j].activate(inputs);
						outputs[k] = hidden[k][j].getOutput();
					}						
					for (int k = 0; k < numNodes; k++){
						inputs[k] = outputs[k];
					}
				}
				
				// Output layer (and calculate output delta)
				for (int j = 0; j < numOutputs; j++){
					output[j].activate(inputs);
					finalOut[j] = output[j].getOutput();
					output[j].setDelta(finalOut[j] - expected);
				}
				
				// Begin back-propagation of delta					
				for (int j = 0; j < numNodes; j++){
					sum = 0.0;
					for (int k = 0; k < numOutputs; k++){
						sum += output[k].getDelta() * output[k].getWeight()[j];
					}
					hidden[j][numLayers - 1].setDelta(hidden[j][numLayers - 1].getDerivative() * sum);
				}			
							
				for (int j = numLayers - 2; j >= 0; j--){
					for(int k = 0; k < numNodes; k++){
						sum = 0.0;
						if(j > 0){
							for (int l = 0; l < numNodes; l++){
								sum += hidden[l][j + 1].getDelta() * hidden[l][j + 1].getWeight()[k]; 
							}
						}
						else{
							for (int l = 0; l < numInputs; l++){
								sum += hidden[l][j + 1].getDelta() * hidden[l][j + 1].getWeight()[k]; 
							}
						}
						hidden[k][j].setDelta(sum);
					}
				}
				
				// Begin forward propagation of weights
				for (int j = 0; j < numInputs; j++){	
					firstIn[j] = trainSet[j][i];
				}
				
				Neuron n;
				double wPrime;				
				for (int j = 0; j < numLayers; j++){
					if(j == 0){
						for (int k = 0; k < numInputs; k++){
							n = hidden[k][j];						
							for (int l = 0; l < numInputs; l++){
								wPrime = n.getWeight()[l] - rate * n.getDelta() * n.getDerivative() * firstIn[l];
								n.setWeight(l, wPrime);
							}
						}
						for (int k = 0; k < numInputs; k++){
							inputs[k] = hidden[k][j].getOutput();
						}
					}
					else{
						for (int k = 0; k < numNodes; k++){
							n = hidden[k][j];						
							for (int l = 0; l < numNodes; l++){
								wPrime = n.getWeight()[l] - rate * n.getDelta() * n.getDerivative() * inputs[l];
								n.setWeight(l, wPrime);
							}
						}
						for (int k = 0; k < numNodes; k++){
							inputs[k] = hidden[k][j].getOutput();
						}
					}
				}
				
				for (int j = 0; j < numOutputs; j++){
					n = output[j];
					for (int k = 0; k < numNodes; k++){
						wPrime = n.getWeight()[k] - rate * n.getDelta() * n.getDerivative() * inputs[k];
						n.setWeight(k, wPrime);
					}
				}
				
				a++;
				if(a == epochs)
					break;
			}
		}
		
	}

}
