package NerualNet;

public class ConsequentNeuron extends Neuron{

	double[] params;
	
	
	ConsequentNeuron(int numInputs, int numParams) {
		super(numInputs, 1);
		this.params = new double[numParams];
	}
	
	protected void activate(double[] inputs, double weight){
		double sum = 0;
		for(int i=0; i < this.params.length; i++){
			sum += params[i] * inputs[i];
		}
		
		this.output = weight * sum;
	}
	
	public double[] getParams() {
		return params;
	}

	public void setParams(double[] params) {
		this.params = params;
	}
}
