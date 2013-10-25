package NerualNet;

import java.util.Random;

public class MultiplierNeuron extends Neuron{
	
	double w;
	
	MultiplierNeuron(int numInputs) {
		super(numInputs, 1);
		// TODO Auto-generated constructor stub
	}

	protected void activate(double[] inputs){
		double product = 1.0;
		
		for(int i = 0; i < this.numInputs; i++){
			product *= inputs[i];
		}
		this.output = product;
	}
	
	public double getW() {
		return w;
	}

	public void setW(double w) {
		this.w = w;
	}
}
