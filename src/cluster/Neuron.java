package cluster;

// Implements a Neuron class used to build a MLPNN
public class Neuron {
	private int numInputs;
	private double output = 0.0;
	private double[] weight;
	
	// Constructor
	Neuron(int numInputs){
		this.numInputs = numInputs;
		this.weight = new double[this.numInputs];
	}
	
	// Activate the neuron for some input
	protected void activate(double[] inputs){
		double sum = 0.0;
		for(int i = 0; i < this.numInputs; i++){
			sum += Math.pow((this.weight[i] - inputs[i]), 2.0); 
		}
		this.output = Math.sqrt(sum);
	}
	
// Getters and Setters ///////////////////////////////
	protected double getOutput(){
		return this.output;
	}
	
	protected void setWeight(double[] wPrime){
		System.arraycopy(wPrime, 0, this.weight, 0, wPrime.length);
	}
	
	protected double[] getWeight(){
		return this.weight;
	}
}
