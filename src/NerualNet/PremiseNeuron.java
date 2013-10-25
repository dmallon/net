package NerualNet;


public class PremiseNeuron extends RBFNeuron{
	
	

	double b;
	double c;
	
	PremiseNeuron(int numInputs) {
		super(numInputs);
	}
	
	
	// Membership Function
	protected void activate(double x){
		this.output = 1/(1+Math.pow(Math.abs((x-this.c)/this.spread), 2*this.b));
	}


	public double getDerivative() {
		// TODO Auto-generated method stub
		return 1.0;
	}
	
	public double getB() {
		return b;
	}

	public double getC() {
		return c;
	}


	public void setC(double c) {
		this.c = c;
	}


	public void setB(double b) {
		this.b = b;
	}
}
