package NerualNet;

import java.util.Random;

public class RBFNet implements Network {
	private int numInputs;
	private int numOutputs;
	private int numCenters;
	
	private double rate;
	private double max;
	
	private Neuron[] output;
	private RBFNeuron[] hidden;
	
	private Random rnd = new Random();
	
	
	RBFNet(int inputs, int centers, int outputs, double rate, int numSamples, double[][] set){
		this.numInputs = inputs;
		this.numCenters = centers;
		this.numOutputs = outputs;
		this.rate = rate;
		this.max = 0.0;
		
		this.hidden = new RBFNeuron[this.numCenters];
		this.output = new Neuron[this.numOutputs];
		
		int index;
		double[] input = new double[this.numInputs];
		
		// Initialize hidden nodes
		for (int i = 0; i < this.numCenters; i++){
			this.hidden[i] = new RBFNeuron(this.numInputs);
		}
		// Initialize output nodes
		for (int i = 0; i < this.numOutputs; i++){
			this.output[i] = new Neuron(this.numCenters, 1);
		}
		
		this.setMaxD(set, numSamples);
		
		for (int i = 0; i < this.numCenters; i++){
			index = rnd.nextInt(numSamples);
			
			for (int j = 0; j < this.numInputs; j++){
				input[j] = set[j][index];
			}			
			this.hidden[i].setCenter(input);
			
			this.hidden[i].setSpread(this.max/Math.sqrt(this.numCenters));
		}	
	}
	
	private void setMaxD(double[][] set, int numSamples){
		double d = 0.0;
		
		for (int i = 0; i < numSamples - 1; i++){
			for (int j = i + 1; j < numSamples; j++){
				for (int k = 0; k < this.numInputs; k++){
					d += Math.pow((set[k][i] - set[k][j]), 2.0);
				}
				d = Math.sqrt(d);
				this.max = Math.max(this.max, d);
			}
		}
	}
	
	public void train(double[][] set, int numSamples, int epochs){		
		int a = 0;
		double wPrime;
		double[] input = new double[this.numInputs];
		double[] expected = new double[this.numOutputs];
		double[] hiddenOut = new double[this.numCenters];
		double[] error = new double[this.numOutputs];
		double[] error2 = new double[this.numCenters];
		double[] uPrime = new double[this.numInputs];
			
		
		while (a < epochs){
			for (int i = 0; i < numSamples; i++){
				this.rate = (this.rate / 1.000001);
				for (int j = this.numInputs; j < this.numInputs + this.numOutputs; j++){
					expected[j - this.numInputs] = set[j][i];
				}
				
				// Activate the hidden nodes
				for (int j = 0; j < this.numCenters; j++){
					for (int k = 0; k < this.numInputs; k++){
						input[k] = set[k][i];
					}				
					this.hidden[j].activate(input);
					hiddenOut[j] = this.hidden[j].getOutput();
				}
				
				// Activate the output node				
				for (int j = 0; j < this.numOutputs; j++){
					this.output[j].activate(hiddenOut);
					error[j] = this.output[j].getOutput() - expected[j];
				}
				
				Neuron n;
				
				for (int j = 0; j < this.numOutputs; j++){
					n = this.output[j];
					for (int k = 0; k < this.numCenters; k++){
						wPrime = n.getWeight()[k] - this.rate * error[j] * hiddenOut[k];
						n.setWeight(k, wPrime);
					}
				}
				
				double sum;
				
				for (int j = 0; j > this.numCenters; j++){
					sum = 0.0;
					for (int k = 0; k < this.numOutputs; k++){
						sum += this.output[k].getWeight()[j] * error[k];
					}			
					error2[j] = sum;
				}
				
				for (int j = 0; j < this.numCenters; j++){
					for (int k = 0; k < this.numInputs; k++){
						uPrime[k] = this.hidden[j].getCenter()[k] - this.rate * error2[j] * set[k][i];
					}
					this.hidden[j].setCenter(uPrime);
					this.hidden[j].setSpread(this.hidden[j].getSpread() - this.rate * error2[j]);
				}
				
				a++;
				if(a == epochs)
					break;
				//System.out.println("\n");
			}
		}
	}	
	
	public double process(double[][] set, int numSamples){
		double totalE = 0.0;
		double[] input = new double[this.numInputs];
		double[] expected = new double[this.numOutputs];
		double[] error = new double[this.numOutputs];
		double[] hiddenOut = new double[this.numCenters];
		
		
		for (int i = 0; i < numSamples; i++){
			
			for (int j = 0; j < this.numInputs; j++){
				input[j] = set[j][i];
			}

			for (int j = this.numInputs; j < this.numInputs + this.numOutputs; j++){
				expected[j - this.numInputs] = set[j][i];
			}
			
			for (int j = 0; j < this.numCenters; j++){				
				this.hidden[j].activate(input);
				hiddenOut[j] = this.hidden[j].getOutput();
			}
			
			// Activate the output node			
			for (int j = 0; j < this.numOutputs; j++){
				this.output[j].activate(hiddenOut);
				error[j] = this.output[j].getOutput() - expected[j];
				totalE += (Math.abs(error[j]/expected[j]));
			}
		}
		return ((totalE/numSamples)*100);
	}
}
