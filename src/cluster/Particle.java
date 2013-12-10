package cluster;

public class Particle {
	private double[] position;
	private double[] velocity;
	private double[] pBest;
	private double fitness;
	
	private int numInputs;
	private int numClusters;
	
	public Particle(int numInputs, int numClusters){
		this.numInputs = numInputs;
		this.numClusters = numClusters;
		this.fitness = 10000000000.0;
		
		this.position = new double[this.numInputs * this.numClusters];
		this.velocity = new double[this.numInputs * this.numClusters];
		this.pBest = new double[this.numInputs * this.numClusters];
	}
	
	public double[] getPosition(){
		return this.position;
	}
	
	public double[] getVelocity(){
		return this.velocity;
	}
	
	public double[] getPBest(){
		return this.pBest;
	}
	
	public double getFitness(){
		return this.fitness;
	}
	
	public void setPosition(double[] pos){
		System.arraycopy(pos, 0, this.position, 0, pos.length);
	}
	
	public void setVelocity(double[] vel){
		System.arraycopy(vel, 0, this.velocity, 0, vel.length);
	}
	
	public void setPBest(double[] pBest){
		System.arraycopy(pBest, 0, this.pBest, 0, pBest.length);
	}
	
	public void setFitness(double fitness){
		this.fitness = fitness;
	}
}
