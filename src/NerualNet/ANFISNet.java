package NerualNet;

import java.util.Random;

public class ANFISNet implements Network {
	private int numInputs;
	private int numLabels;
	private int numMF;
	private int numRules;
	private int numOutputs;
	
	private char classifier;
	
	private double rate;
	private double max;
	
	private PremiseNeuron[] premiseLayer;
	private MultiplierNeuron[] multiplierLayer;
	private NormalizerNeuron[] normalizerLayer;
	private ConsequentNeuron[] consequentLayer;
	private Neuron[] outputLayer;
	
	private Random rnd = new Random();
	
	/**
	 * 
	 * @param inputs
	 * @param numLabels
	 * @param outputs
	 * @param rate
	 * @param classifier
	 * @param numSamples
	 * @param set
	 */
	public ANFISNet(int inputs, int numLabels, int outputs, double rate, char classifier, int numSamples, double[][] set){		
		this.numInputs = inputs;
		this.numMF = inputs * numLabels;
//		this.numRules = (int) Math.pow(numLabels, inputs);
		this.numRules = numLabels;
		this.numLabels = numLabels;
		this.numOutputs = outputs;
		this.classifier = classifier;
		this.rate = rate;
		this.max = 0.0;
		
		this.premiseLayer = new PremiseNeuron[this.numMF];
		this.multiplierLayer = new MultiplierNeuron[this.numRules];
		this.normalizerLayer = new NormalizerNeuron[this.numRules];
		this.consequentLayer = new ConsequentNeuron[this.numRules];
		this.outputLayer = new Neuron[this.numOutputs];
		
		
		
		// Initialize premise layer
		for (int i = 0; i < this.numMF; i++){
			this.premiseLayer[i] = new PremiseNeuron(1);
		}
		// Initialize multiplier layer
		for (int i = 0; i < this.numRules; i++){
			this.multiplierLayer[i] = new MultiplierNeuron(this.numInputs);
		}
		// Initialize normalization layer
		for (int i = 0; i < this.numRules; i++){
			this.normalizerLayer[i] = new NormalizerNeuron(this.numRules);
		}
		// Initialize consequent layer
		for (int i = 0; i < this.numRules; i++){
			this.consequentLayer[i] = new ConsequentNeuron(this.numRules, this.numInputs);
		}
		// Initialize output nodes
		for (int i = 0; i < this.numOutputs; i++){
			this.outputLayer[i] = new Neuron(this.numRules, 1);
		}
		
		this.setMaxD(set, numSamples);
		
		int index;
		double[] input = new double[this.numInputs];
		
		// Initialize premise layer params
		for (int i = 0; i < this.numMF; i++){
			index = rnd.nextInt(numSamples);
			
			for (int j = 0; j < this.numInputs; j++){
				input[j] = set[j][index];
			}			
			this.premiseLayer[i].setCenter(input);
			
			this.premiseLayer[i].setSpread(this.max/Math.sqrt(this.numMF));
		}	
	}
	
	
	/**
	 * Set maximum distance between membership functions
	 * 
	 * @param set
	 * @param numSamples
	 */
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
		double expected;
		double firstIn;
		double[] inputs = new double[this.numInputs];
		double[] multiplierInputs = new double[this.numInputs];
		double[] normalizerInputs = new double[this.numRules];
		double[] premiseOut = new double[this.numMF];
		double[] multiplierOut = new double[this.numRules];
		double[] normalizerOut = new double[this.numRules];
		double[] consequentOut = new double[this.numRules];
		double[] finalOut = new double[this.numOutputs];
		double[] error2 = new double[this.numMF];
		double uPrime;
		double bPrime;
		
			
		
		while (a < epochs){
			for (int i = 0; i < numSamples; i++){
				this.rate = (this.rate / 1.000001);
				if(this.classifier == classes[i])
					expected = 1.0;
				else
					expected = 0.0;
				
				// Set up input vector
				for (int k = 0; k < this.numInputs; k++){
					inputs[k] = set[k][i];
				}
				
				// Activate the premise nodes
				int x = 0; // input index
				for (int j = 0; j < this.numMF; j++){
					
					// feed input to each A, B, C .. #MF
					firstIn = inputs[x];
					if(j % this.numLabels == 0 && j != 0){
						x++;
					}
					
					this.premiseLayer[j].activate(firstIn);
					premiseOut[j] = this.premiseLayer[j].getOutput();
				}
				
				// Activate the multiplier nodes
				for (int j = 0; j < this.numRules; j++){
					for (int k = 0, m = 0; k < this.numMF; k += this.numLabels, m++){
						multiplierInputs[m] = this.premiseLayer[j].getOutput();
					}
					this.multiplierLayer[j].activate(multiplierInputs);
					multiplierOut[j] = this.multiplierLayer[j].getOutput();
				}
				
				// Activate the normalizer nodes
				for (int j = 0; j < this.numRules; j++){
					for (int k = 0; k < this.numRules; k++){
						normalizerInputs[k] = this.multiplierLayer[k].getOutput();
					}
					this.normalizerLayer[j].activate(normalizerInputs);
					normalizerOut[j] = this.normalizerLayer[j].getOutput();
				}
				
				// Activate the consequent nodes
				for (int j = 0; j < this.numRules; j++){
					this.consequentLayer[j].activate(inputs, this.consequentLayer[j].getWeight()[0]); // Get first (and only) weight for consequent neuron
					consequentOut[j] = this.consequentLayer[j].getOutput();
				}
				
				
				// Output layer (and calculate output delta)
				for (int j = 0; j < this.numOutputs; j++){
					this.outputLayer[j].activate(inputs);
					finalOut[j] = this.outputLayer[j].getOutput();
					
					// Calculate output layer delta
					this.outputLayer[j].setDelta(finalOut[j] - expected);
				}
				
				
				/*****************************************
				 * 
				 *  Begin back-propagation
				 *  
				 *****************************************/
				
				// Calculate consequent layer deltas
				double sum;
				for (int j = 0; j < this.numRules; j++){
					sum = 0.0;
					for (int k = 0; k < this.numOutputs; k++){
						sum += this.outputLayer[k].getDelta() * this.outputLayer[k].getWeight()[j];
					}
					this.consequentLayer[j].setDelta(this.consequentLayer[j].getDerivative() * sum);
				}
				
				// Calculate normalizer layer deltas
				for (int j = 0; j < this.numRules; j++){
					sum = 0.0;
					sum += this.consequentLayer[j].getDelta() * this.consequentLayer[j].getWeight()[j];
					
					this.normalizerLayer[j].setDelta(this.normalizerLayer[j].getDerivative() * sum);
				}
				
				// Calculate multiplier layer deltas
				for (int j = 0; j < this.numRules; j++){
					sum = 0.0;
					for (int k = 0; k < this.numRules; k++){
						sum += this.normalizerLayer[k].getDelta() * this.normalizerLayer[k].getWeight()[j];
					}
					this.multiplierLayer[j].setDelta(this.multiplierLayer[j].getDerivative() * sum);
				}
				
				
				
				// Update premise layer's parameters
				for (int j = 0; j < this.numRules; j++){
					uPrime = this.premiseLayer[j].getC() - this.rate * error2[j] * set[j][i];
					bPrime = this.premiseLayer[j].getB() - this.rate * error2[j] * set[j][i];
					
					this.premiseLayer[j].setC(uPrime);
					this.premiseLayer[j].setSpread(this.premiseLayer[j].getSpread() - this.rate * error2[j]);
					this.premiseLayer[j].setB(bPrime);
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
		double[] hiddenOut = new double[this.numMF];
		double out = 0.0;
		
		for (int j = 0; j < this.numInputs; j++){
			input[j] = set[j][index];
		}
		
		for (int j = 0; j < this.numMF; j++){				
			this.premiseLayer[j].activate(input);
			hiddenOut[j] = this.premiseLayer[j].getOutput();
		}
		
		// Activate the output node			
		for (int j = 0; j < this.numOutputs; j++){
			this.outputLayer[j].activate(hiddenOut);
			if(this.outputLayer[j].getOutput() > 0.5)
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
