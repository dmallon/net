package NerualNet;

import java.util.Random;

public class MultiplierNeuron extends Neuron{
	
	MultiplierNeuron(int numInputs) {
		super(numInputs);
		// TODO Auto-generated constructor stub
	}

	protected void activate(double[] inputs){
		double product = 1.0;
		
		for(int i = 0; i < this.numInputs; i++){
			product *= inputs[i];
		}
		this.output = product;
	}
}
