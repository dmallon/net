package cluster;

// Class to define MLPNN structure and functionality
public class CompNet{	
	protected int numInputs;
	protected int numLayers;
	protected int numOutputs;
	protected int numNodes;
	protected double threshold = 0.65;
	protected char[] classes;
	
	protected double rate;
	
	protected Neuron[] output;
	protected Neuron[][] hidden;
	
	// Constructor 
	CompNet(int inputs, int layers, int outputs, double rate, char[] classes){	
		this.numInputs = inputs;
		this.numLayers = layers;
		this.numOutputs = outputs;
		this.numNodes = inputs;
		this.rate = rate;
		this.classes = classes;
		
		// Initialize Net structure
		this.hidden = new Neuron[this.numNodes][this.numLayers];
		this.output = new Neuron[this.numOutputs];
		
		// Fill the hidden layers with initialized nodes
		for (int i = 0; i < this.numLayers; i++){
			for (int j = 0; j < this.numNodes; j++){
				if(i == 0)
					this.hidden[j][i] = new Neuron(numInputs, 2);
				else
					this.hidden[j][i] = new Neuron(numNodes, 2);
			}
		}
		// Fill the output layer with initialized nodes
		for (int j = 0; j < this.numOutputs; j++){
			this.output[j] = new Neuron(numNodes, 2);
		}
	}

	
	// Pass training call to desired strategy
	public void train(double[][] trainSet, int numSamples, int epochs, char[] classes){
		double expected;
		double[] firstIn = new double[this.numInputs];
		double[] inputs = new double[this.numNodes];
		double[] outputs = new double[this.numNodes];
		double[] finalOut = new double[this.numOutputs];
		double sum;
		
		int a = 0;
		// Loop for all epochs
		while (a < epochs){
			// Loop through all samples
			for (int i = 0; i < numSamples; i++){
				
				// Get the input from the current training vector
				for (int j = 0; j < this.numInputs; j++){
					firstIn[j] = trainSet[j][i];
				}
				
				// Check the expected output
				if(this.classes == classes[i])
					expected = 1.0;
				else
					expected = 0.0;
				
				// Activate hidden layers
				for (int j = 0; j < this.numLayers; j++){
					for (int k = 0; k < this.numNodes; k++){
						if(j == 0)
							this.hidden[k][j].activate(firstIn);
						else
							this.hidden[k][j].activate(inputs);
						outputs[k] = this.hidden[k][j].getOutput();
					}						
					for (int k = 0; k < this.numNodes; k++){
						inputs[k] = outputs[k];
					}
				}
				
				// Activate Output layer (and calculate output delta)
				for (int j = 0; j < this.numOutputs; j++){
					this.output[j].activate(inputs);
					finalOut[j] = this.output[j].getOutput();
					this.output[j].setDelta(finalOut[j] - expected);
				}
				
				// Begin back-propagation of delta at Output nodes				
				for (int j = 0; j < this.numNodes; j++){
					sum = 0.0;
					for (int k = 0; k < this.numOutputs; k++){
						sum += this.output[k].getDelta() * this.output[k].getWeight()[j];
					}
					this.hidden[j][this.numLayers - 1].setDelta(this.hidden[j][this.numLayers - 1].getDerivative() * sum);
				}			
				// Backprop through hidden nodes			
				for (int j = this.numLayers - 2; j >= 0; j--){
					for(int k = 0; k < this.numNodes; k++){
						sum = 0.0;
						if(j > 0){
							for (int l = 0; l < this.numNodes; l++){
								sum += this.hidden[l][j + 1].getDelta() * this.hidden[l][j + 1].getWeight()[k]; 
							}
						}
						else{
							for (int l = 0; l < this.numInputs; l++){
								sum += this.hidden[l][j + 1].getDelta() * this.hidden[l][j + 1].getWeight()[k]; 
							}
						}
						this.hidden[k][j].setDelta(sum);
					}
				}
				
				// Begin forward propagation of weights
				for (int j = 0; j < this.numInputs; j++){	
					firstIn[j] = trainSet[j][i];
				}
				
				Neuron n;
				double wPrime;
				// Set the weights for the hidden layers
				for (int j = 0; j < this.numLayers; j++){
					if(j == 0){
						for (int k = 0; k < this.numInputs; k++){
							n = this.hidden[k][j];						
							for (int l = 0; l < this.numInputs; l++){
								wPrime = n.getWeight()[l] - this.rate * n.getDelta() * n.getDerivative() * firstIn[l];
								n.setWeight(l, wPrime);
							}
						}
						for (int k = 0; k < this.numInputs; k++){
							inputs[k] = this.hidden[k][j].getOutput();
						}
					}
					else{
						for (int k = 0; k < this.numNodes; k++){
							n = this.hidden[k][j];						
							for (int l = 0; l < this.numNodes; l++){
								wPrime = n.getWeight()[l] - this.rate * n.getDelta() * n.getDerivative() * inputs[l];
								n.setWeight(l, wPrime);
							}
						}
						for (int k = 0; k < this.numNodes; k++){
							inputs[k] = this.hidden[k][j].getOutput();
						}
					}
				}
				// Set the weights for the output layer
				for (int j = 0; j < this.numOutputs; j++){
					n = this.output[j];
					for (int k = 0; k < this.numNodes; k++){
						wPrime = n.getWeight()[k] - this.rate * n.getDelta() * n.getDerivative() * inputs[k];
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
	
	// Process to the test set to produce final output
	public int process(double[][] set, int index){
		double[] firstIn = new double[this.numInputs];
		double[] inputs = new double[this.numNodes];
		double[] outputs = new double[this.numNodes];
		double[] finalOut = new double[this.numOutputs];
		double out = 0.0;
		
		// Get the input from the set
		for (int j = 0; j < this.numInputs; j++){
			firstIn[j] = set[j][index];
		}
		
		// Hidden layers
		for (int j = 0; j < this.numLayers; j++){
			for (int k = 0; k < this.numNodes; k++){
				if(j == 0)
					this.hidden[k][j].activate(firstIn);
				else
					this.hidden[k][j].activate(inputs);
				outputs[k] = this.hidden[k][j].getOutput();
			}						
			for (int k = 0; k < this.numNodes; k++){
				inputs[k] = outputs[k];
			}
		}
		
		// Output layer (and calculate output delta)
		for (int j = 0; j < this.numOutputs; j++){
			this.output[j].activate(inputs);
			finalOut[j] = this.output[j].getOutput();
			//System.out.println(this.classifier + " " + finalOut[j]);
			if(finalOut[j] > threshold)
				out = 1.0;
			else
				out = 0.0;					
		}
		
		// Return the results
		if(out == 1.0)
			return (this.classes[0]);
		else
			return '!';
	}	
}
