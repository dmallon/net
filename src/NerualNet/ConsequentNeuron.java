package NerualNet;

public class ConsequentNeuron extends Neuron{

	ConsequentNeuron(int numInputs) {
		super(numInputs);
		// TODO Auto-generated constructor stub
	}
	
	protected void activate(double[] inputs){
		double sum = 0.0;
		for(int i = 0; i < this.numInputs; i++){
			sum += this.weight[i]*inputs[i]; 
		}
	}
}
