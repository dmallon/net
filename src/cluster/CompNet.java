package cluster;

// Class to define MLPNN structure and functionality
public class CompNet{	
	protected int numInputs;
	protected int numOutputs;
	protected double threshold = 0.65;
	
	protected double rate;
	
	protected Neuron[] output;
	protected Neuron[][] hidden;
	
	// Constructor 
	CompNet(int inputs, int outputs, double rate){	 
		this.numInputs = inputs;
		this.numOutputs = outputs;
		this.rate = rate;
		
		// Initialize Net structure
		this.output = new Neuron[this.numOutputs];
		
		// Fill the output layer with initialized nodes
		for (int j = 0; j < this.numOutputs; j++){
			this.output[j] = new Neuron(numInputs);
		}
	}

	
	// Pass training call to desired strategy
	public void train(double[][] trainSet, int numSamples, int epochs, char[] classifier){
		double[] input = new double[this.numInputs];
		double[] wPrime = new double[this.numInputs];
		double maxOutput;
		int maxIndex;
		
		int a = 0;
		// Loop for all epochs
		while (a < epochs){
			// Loop through all samples
			for (int i = 0; i < numSamples; i++){
				
				// Get the input from the current training vector
				for (int j = 0; j < this.numInputs; j++){
					input[j] = trainSet[j][i];
				}
				
				// Activate Output layer (and calculate output delta)
				maxOutput = 0.0;
				maxIndex = 0;
				for (int j = 0; j < this.numOutputs; j++){
					this.output[j].activate(input);
					if(this.output[j].getOutput() > maxOutput){
						maxOutput = this.output[j].getOutput();
						maxIndex = j;
					}
				}
				
				// Set the weights for max output node
				Neuron n = this.output[maxIndex];
				for (int k = 0; k < this.numInputs; k++){
					wPrime[k] = n.getWeight()[k] + this.rate * (input[k] - n.getWeight()[k]);
				}
				n.setWeight(wPrime);
				
				// Increment epoch counter
				a++;
				// Exit loop if number of epochs has been reached
				if(a == epochs)
					break;
			}
		}			
	}
	
	// Process to the test set to produce final output
	public void process(double[][] testSet, int numSamples, char[] classes){
		double[] input = new double[this.numInputs];
		double maxOutput;
		int maxIndex;
		
		for(int i = 0; i < numSamples; i++){
			// Get the input from the set
			for (int j = 0; j < this.numInputs; j++){
				input[j] = testSet[j][i];
			}
			
			// Output layer
			maxOutput = 0.0;
			maxIndex = 0;
			for (int j = 0; j < this.numOutputs; j++){
				this.output[j].activate(input);	
				if(this.output[j].getOutput() > maxOutput){
					maxOutput = this.output[j].getOutput();
					maxIndex = j;
				}
			}
			System.out.println(maxIndex + ", " + classes[i]);
		}
	}	
}
