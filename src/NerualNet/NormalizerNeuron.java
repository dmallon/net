package NerualNet;

public class NormalizerNeuron extends Neuron{
	
	NormalizerNeuron(int numInputs) {
		super(numInputs, 1);		
	}

	protected void activate(double[] inputs){
		
		for(int i = 0; i < this.numInputs; i++){
//			product *= inputs[i]; 
		}
	}
	
}
