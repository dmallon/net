package NerualNet;

import java.util.Random;

public class ANFISNet implements Network {
	private int numInputs;
	private int numLabels;
	private int numCenters;
	private int numRules;
	private int numOutputs;
	
	private char classifier;
	
	private double rate;
	private double max;
	
	private PremiseNeuron[] premiseLayer;
	private MultiplierNeuron[] multiplierLayer;
	private NormalizerNeuron[] normalizationLayer;
	private ConsequentNeuron[] consequentLayer;
	private Neuron[] output;
	
	private Random rnd = new Random();
	
	
	public ANFISNet(int inputs, int numLabels, int outputs, double rate, char classifier, int numSamples, double[][] set){
		
		System.out.print(inputs);
		
		this.numInputs = inputs;
		this.numCenters = inputs * numLabels;
		this.numRules = (int) Math.pow(numLabels, inputs);
		this.numLabels = numLabels;
		this.numOutputs = outputs;
		this.classifier = classifier;
		this.rate = rate;
		this.max = 0.0;
		
		this.premiseLayer = new PremiseNeuron[this.numCenters];
		this.multiplierLayer = new MultiplierNeuron[this.numRules];
		this.normalizationLayer = new NormalizerNeuron[this.numRules];
		this.consequentLayer = new ConsequentNeuron[this.numRules];
		this.output = new Neuron[this.numOutputs];
		
		
		int index;
		double[] input = new double[this.numInputs];
		
		// Initialize premise layer
		for (int i = 0; i < this.numCenters; i++){
			this.premiseLayer[i] = new PremiseNeuron(1);
		}
		// Initialize multiplier layer
		for (int i = 0; i < this.numRules; i++){
			this.multiplierLayer[i] = new MultiplierNeuron(this.numLabels);
		}
		// Initialize normalization layer
		for (int i = 0; i < this.numRules; i++){
			this.normalizationLayer[i] = new NormalizerNeuron(this.numRules);
		}
		// Initialize consequent layer
		for (int i = 0; i < this.numRules; i++){
			this.consequentLayer[i] = new ConsequentNeuron(this.numInputs + 1);
		}
		// Initialize output nodes
		for (int i = 0; i < this.numOutputs; i++){
			this.output[i] = new Neuron(this.numRules, 1);
		}
		
		this.setMaxD(set, numSamples);
		
		for (int i = 0; i < this.numCenters; i++){
			index = rnd.nextInt(numSamples);
			
			for (int j = 0; j < this.numInputs; j++){
				input[j] = set[j][index];
			}			
			this.premiseLayer[i].setCenter(input);
			
			this.premiseLayer[i].setSpread(this.max/Math.sqrt(this.numCenters));
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
	
	
	public void train(double[][] set, int numSamples, int epochs, char[] classes){		
		int a = 0;
		double wPrime;
		double expected;
		double[] input = new double[this.numInputs];
		double[] premiseOut = new double[this.numCenters];
		double[] multiplierOut = new double[this.numRules];
		double[] normalizerOut = new double[this.numRules];
		double[] consequentOut = new double[this.numRules];
		double[] error = new double[this.numOutputs];
		double[] error2 = new double[this.numCenters];
		double[] uPrime = new double[this.numInputs];
			
		
		while (a < epochs){
			for (int i = 0; i < numSamples; i++){
				this.rate = (this.rate / 1.000001);
				if(this.classifier == classes[i])
					expected = 1.0;
				else
					expected = 0.0;
				
				// Populate input from set
				for (int k = 0; k < this.numInputs; k++){
					input[k] = set[k][i];
				}
				
				for (int j = 0; j < this.numCenters; j++){
					for (int k = 0; k < this.numInputs; k++){
						input[k] = set[k][i];
					}
					this.premiseLayer[j].activate(input);
					premiseOut[j] = this.premiseLayer[j].getOutput();
				}
				
				// Activate the premise nodes
				for (int j = 0; j < this.numCenters; j++){
					for (int k = 0; k < this.numInputs; k++){
						input[k] = set[k][i];
					}
					this.premiseLayer[j].activate(input);
					premiseOut[j] = this.premiseLayer[j].getOutput();
				}
				
				
				
				// Activate the multiplier nodes
				for (int j = 0; j < this.numCenters; j++){
					for (int k = 0; k < this.numInputs; k++){
						input[k] = set[k][i];
					}
					this.premiseLayer[j].activate(input);
					premiseOut[j] = this.premiseLayer[j].getOutput();
				}
				
				// Activate the normalizer nodes
				for (int j = 0; j < this.numCenters; j++){
					for (int k = 0; k < this.numInputs; k++){
						input[k] = set[k][i];
					}
					this.premiseLayer[j].activate(input);
					premiseOut[j] = this.premiseLayer[j].getOutput();
				}
				
				// Activate the consequent nodes
				for (int j = 0; j < this.numCenters; j++){
					for (int k = 0; k < this.numInputs; k++){
						input[k] = set[k][i];
					}
					this.premiseLayer[j].activate(input);
					premiseOut[j] = this.premiseLayer[j].getOutput();
				}
				
				// Activate the output node				
//				for (int j = 0; j < this.numOutputs; j++){
//					this.output[j].activate(hiddenOut);
//					error[j] = this.output[j].getOutput() - expected;
//				}
//				
//				Neuron n;
//				
//				for (int j = 0; j < this.numOutputs; j++){
//					n = this.output[j];
//					for (int k = 0; k < this.numCenters; k++){
//						wPrime = n.getWeight()[k] - this.rate * error[j] * hiddenOut[k];
//						n.setWeight(k, wPrime);
//					}
//				}
				
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
						uPrime[k] = this.premiseLayer[j].getCenter()[k] - this.rate * error2[j] * set[k][i];
					}
					this.premiseLayer[j].setCenter(uPrime);
					this.premiseLayer[j].setSpread(this.premiseLayer[j].getSpread() - this.rate * error2[j]);
				}
				
				a++;
				if(a == epochs)
					break;
				//System.out.println("\n");
			}
		}
	}	
	
	public char process(double[][] set, int index){
		double[] input = new double[this.numInputs];
		double[] hiddenOut = new double[this.numCenters];
		double out = 0.0;
		
		for (int j = 0; j < this.numInputs; j++){
			input[j] = set[j][index];
		}
		
		for (int j = 0; j < this.numCenters; j++){				
			this.premiseLayer[j].activate(input);
			hiddenOut[j] = this.premiseLayer[j].getOutput();
		}
		
		// Activate the output node			
		for (int j = 0; j < this.numOutputs; j++){
			this.output[j].activate(hiddenOut);
			if(this.output[j].getOutput() > 0.5)
				out = 1.0;
			else
				out = 0.0;
		}
		
		if(out == 1.0)
			return (this.classifier);
		else
			return '!';
		
	}
	
}
