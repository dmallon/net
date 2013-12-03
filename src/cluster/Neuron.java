package cluster;

import java.util.Random;

// Implements a Neuron class used to build a MLPNN
public class Neuron {
	protected int numInputs;
	protected int activation;
	protected double delta;
	protected double output;
	protected double derivative;
	protected double[] weight;
	protected Random rnd;
	
	// Constructor
	Neuron(int numInputs, int activation){
		this.rnd = new Random();
		this.activation = activation;
		this.delta = 0.0;
		this.output = 0.0;
		this.derivative = 0.0;
		this.numInputs = numInputs;
		this.weight = new double[this.numInputs];
		
		// Fill the weight array with random values
		for (int i = 0; i < this.numInputs; i++){
			this.weight[i] = (this.rnd.nextDouble() * 0.6) - 0.3;
		}
	}	
	
	// Activate the neuron for some input
	protected void activate(double[] inputs){
		double sum = 0.0;
		for(int i = 0; i < this.numInputs; i++){
			sum += this.weight[i]*inputs[i]; 
		}
		// For sigmoid activation
		if(activation == 2){
			this.output = this.sigmoidFunc(sum);
			this.derivative = this.sigDerivative(sum);
		}
		// For linear activation
		else{
			this.output = sum;
			this.derivative = 1.0;
		}
	}
	
	// Sigmoid activation function
	private double sigmoidFunc(double x){
		return Math.tanh(10.0*x);
	}
	
	// Sigmoid derivative
	private double sigDerivative(double x){
		return (1.0 - Math.pow(Math.tanh(10.0*x), 2.0));
	}
	
// Getters and Setters ///////////////////////////////
	protected double getOutput(){
		return this.output;
	}
	
	protected double getDerivative(){
		return this.derivative;
	}
	
	protected double[] getWeight(){
		return this.weight;
	}
	
	protected void setWeight(int index, double wPrime){
		this.weight[index] = wPrime;
	}
	
	protected double getDelta(){
		return this.delta;
	}
	
	protected void setDelta(double delta){
		this.delta = delta;
	}
}
