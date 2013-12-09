package cluster;

import java.util.Random;

// Class to define Competitve Learning net structure and functionality
public class CompNet{	
	private int numInputs;
	private int numOutputs;	
	private double rate;	
	private Neuron[] output;	
	private char[] classes;
	
	private Random rnd = new Random();
	
	// Constructor 
	CompNet(int inputs, int outputs, double rate, char[] classes){	 
		this.numInputs = inputs;
		this.numOutputs = outputs;
		this.rate = rate;
		this.classes = classes;
		
		// Initialize Net structure
		this.output = new Neuron[this.numOutputs];
		
		// Fill the output layer with initialized nodes
		for (int j = 0; j < this.numOutputs; j++){
			this.output[j] = new Neuron(numInputs);
		}
	}

	
	// Pass training call to desired strategy
	public void train(double[][] trainSet, int numSamples, int epochs){
		double[] input = new double[this.numInputs];
		double[] wPrime = new double[this.numInputs];
		double[][] weights = new double[this.numOutputs][this.numInputs];
		double maxOutput;
		int maxIndex;
		int index;
		
		// Initialize the weights to random points from the data set
		for (int i = 0; i < this.numOutputs; i++){
			index = rnd.nextInt(numSamples);		
			for(int j = 0; j < this.numInputs; j++){
				weights[i][j] = trainSet[j][index];
			}
			this.output[i].setWeight(weights[i]);
		}
		
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
	public void test(double[][] testSet, int numSamples, char[] classifier){
		double[] input = new double[this.numInputs];
		int[][] results = new int[this.numOutputs][this.numOutputs];
		double maxOutput;
		int maxIndex;
		
		// Initialize result matrix to 0
		for(int i = 0; i < this.numOutputs; i++){
			for (int j = 0; j < this.numOutputs; j++){
				results[i][j] = 0;
			}
		}
		
		// Process each sample in the set
		for(int i = 0; i < numSamples; i++){
			// Get the input from the set
			for (int j = 0; j < this.numInputs; j++){
				input[j] = testSet[j][i];
			}
			
			// Output layer
			maxOutput = 0.0;
			maxIndex = 0;
			// Find the output node with the highest output signal
			for (int j = 0; j < this.numOutputs; j++){
				this.output[j].activate(input);	
				if(this.output[j].getOutput() > maxOutput){
					maxOutput = this.output[j].getOutput();
					maxIndex = j;
				}
			}
			// Assign point i to the cluster for that output
			for(int j = 0; j < this.numOutputs; j++){
				if(classifier[i] == this.classes[j]){
					results[maxIndex][j]++;
				}
			}
		}
		// Print out the cluster structure
		for(int i = 0; i < this.numOutputs; i++){
			System.out.print("Cluster " + (i + 1) + ": ");
			for (int j = 0; j < this.numOutputs; j++){
				System.out.print(results[i][j] + " " + classes[j] + "'s  ");
			}
			System.out.println();
		}
		
		// Calculate the purity for the clustering
		int sum = 0;
		int max;
		for(int i = 0; i < this.numOutputs; i++){
			max = 0;
			for (int j = 0; j < this.numOutputs; j++){
				if(results[i][j] > max)
					max = results[i][j];
			}
			sum += max;
		}
		System.out.println("Cluster Purity: " + ((double)sum/(double)numSamples));
		
	}	
}
