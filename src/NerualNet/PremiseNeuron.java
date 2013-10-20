package NerualNet;


public class PremiseNeuron extends RBFNeuron{
	
//	double b;
//	double c;
	
	PremiseNeuron(int numInputs) {
		super(numInputs);
	}
	
	
	// Bell curve
//	protected void activate(double x){
//		this.output = 1/(1+Math.pow(Math.abs((x-this.c)/this.spread), 2*this.b));
//	}
}
