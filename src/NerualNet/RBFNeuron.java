package NerualNet;


public class RBFNeuron {
	protected int numInputs;
	protected double output;
	protected double spread;
	protected double[] center;
	
	RBFNeuron(int numInputs){
		this.numInputs = numInputs;
		this.center = new double[this.numInputs];
	}
	
	protected void activate(double[] input){
		this.output = this.gaussian(input);
	}	
	
	protected double getOutput(){
		return this.output;
	}
	
	protected void setCenter(double[] center){
		System.arraycopy(center, 0, this.center, 0, this.numInputs);
	}
	
	protected void setSpread(double spread){
		this.spread = spread;
	}
	
	protected double[] getCenter(){
		return this.center;
	}
	
	protected double getSpread(){
		return this.spread;
	}
	
	private double gaussian(double[] input){
		double exp;		
		exp = -(this.sqrDistance(input) / (2.0 * Math.pow(this.spread, 2.0)));		
		return Math.pow(Math.E, exp);
	}
	
	private double sqrDistance(double[] x){
		double sum = 0.0;
		for (int i = 0; i < this.numInputs; i++){
			sum += Math.pow((x[i] - center[i]), 2.0);
		}
		return sum;
	}
}
