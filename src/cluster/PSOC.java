package cluster;

import java.util.Random;

// Particle Swarm Optimization Cluster Hybrid with K-Means
public class PSOC {
	// Instance vars
	private double[] gBest;
	private Particle[] swarm;
	private char[] classes;
	private double gbFitness;
	private double r1;
	private double r2;
	private int numInputs;
	private int numClusters;
	private Random rnd;
	
	// Tunable parameters
	private int epochs = 500;
	private int numParticles = 10;
	private double weight = 0.72;
	private double c1 = 1.49;
	private double c2 = 1.49;	
	
	// Constructor for PSOC algorithm
	public PSOC(int numInputs, int numClusters, char[] classes){
		this.numInputs = numInputs;
		this.numClusters = numClusters;
		this.classes = classes;
		this.rnd = new Random();
		
		this.gBest = new double[this.numInputs * this.numClusters];
		
		// Initialize and fill the swarm
		this.swarm = new Particle[this.numParticles];
		for (int i = 0; i < this.numParticles; i++){
			this.swarm[i] = new Particle(this.numInputs, this.numClusters);		
		}
		
	}
	
	// Initiate clustering with the PSOC algorithm
	public void cluster(double[][] set, int numSamples, char[] classifier){
		// Lots of local vars
		double[][] centers;
		double[] sample = new double[this.numInputs];
		int[] clusterLbl = new int[numSamples];
		double[] position = new double[this.numInputs * this.numClusters];
		double[] velocity = new double[this.numInputs * this.numClusters];
		KMeans km;
		int index;
		double[] sums = new double[this.numClusters];
		int[] memberCount = new int[this.numClusters];
		boolean loop = true;
		int[][] results = new int[this.numClusters][this.numClusters];
		
		// Initialize the results matrix to 0
		for(int i = 0; i < this.numClusters; i++){
			for (int j = 0; j < this.numClusters; j++){
				results[i][j] = 0;
			}
		}
		// Initialize global fitness
		gbFitness = 100000000.0;
		
		// Use K-Means to set the center of the first particle (hybrid approach)
		km = new KMeans(this.numInputs, this.numClusters, this.classes);
		km.train(set, numSamples);
		centers = km.getCenters();
		
		for(int i = 0; i < this.numClusters; i++){
			for(int j = 0; j < this.numInputs; j++){
				position[(i+1)*j] = centers[i][j];
			}
		}
		
		this.swarm[0].setPosition(position);
		
		// Initialize other particles to random vectors from set
		for(int i = 1; i < this.numParticles; i++){
			for (int j = 0; j < this.numClusters; j++){
				index = rnd.nextInt(numSamples);		
				for(int k = 0; k < this.numInputs; k++){
					position[(j+1)*k] = set[k][index];
				}
			}
			this.swarm[i].setPosition(position);
		}
		
		// Main loop for gBest PSOC Algorithm
		do{
			for(int i = 0; i < this.numParticles; i++){
				// Clear out the sums and member count arrays
				for(int j = 0; j < this.numClusters; j++){
					sums[j] = 0.0;
					memberCount[j] = 0;
				}
				
				// Process each sample in the set
				for(int j = 0; j < numSamples; j++){
					// Get sample j from the set
					for(int k = 0; k < this.numInputs; k++){
						sample[k] = set[k][j];
					}
					// Set the cluster label to the cluster with the minDist from sample
					clusterLbl[j] = this.getLabel(sample, swarm[i].getPosition());			
					sums[clusterLbl[j]] += this.distance(sample, position, clusterLbl[j]);
					memberCount[clusterLbl[j]]++;					
				}
				
				// Calculate the fitness as quantization error
				double fitness = 0.0;
				for (int j = 0; j < this.numClusters; j++){
					if(memberCount[j] != 0)
						fitness += (sums[j])/((double)memberCount[j]);
				}
				fitness = fitness/this.numClusters;
				
				// Set local best for particle
				if(fitness < swarm[i].getFitness()){
					swarm[i].setPBest(position);
					swarm[i].setFitness(fitness);
				}
				
				// Set Global Best
				if(fitness < this.gbFitness){
					System.arraycopy(position, 0, gBest, 0, position.length);
					this.gbFitness = fitness;
				}
				// End the loop if there is no change in global best
				else if((fitness - this.gbFitness) < 0.0000001)
					loop = false;				
				
				// Calculate new velocity and position
				for (int j = 0; j < (this.numInputs * this.numClusters); j++){
					r1 = rnd.nextDouble();
					r2 = rnd.nextDouble();
					// Velocity
					velocity[j] = weight*this.swarm[i].getVelocity()[j] + c1*r1*(this.swarm[i].getPBest()[j] - this.swarm[i].getPosition()[j])
						+ c2*r2*(this.gBest[j] - this.swarm[i].getPosition()[j]);
					
					//Position
					position[j] = this.swarm[i].getPosition()[j] + velocity[j];
				}
				this.swarm[i].setVelocity(velocity);
				this.swarm[i].setPosition(position);
			}
			System.out.println("Global Best Error: " + this.gbFitness);
		}while(loop);
		
				
	}
	
	// Return cluster label of nearest cluster center
	private int getLabel(double[] sample, double[] position){
		double minDist = 100000000.0;
		int minLbl = 0;
		double sum = 0.0;
		double dist;
		
		// Check the distance to each cluster
		for(int a = 0; a < this.numClusters; a++){			
			for(int i = 0; i < this.numInputs; i++){
				sum += Math.pow((position[(a*this.numInputs)+i] - sample[i]), 2.0);
			}
			dist = Math.sqrt(sum);			
			if (Double.isNaN(dist))
				dist = 0.0;
			
			if(dist < minDist){
				minDist = dist;
				minLbl = a;
			}
		}
		return minLbl;
	}
	
	// Calculates Euclidean distance
	private double distance(double[] sample, double[] position, int clusterLbl){
		double sum = 0.0;
		double dist;
		
		for(int i = 0; i < this.numInputs; i++){
			sum += Math.pow((position[(clusterLbl * this.numInputs) + i] - sample[i]), 2.0);
		}
		dist = Math.sqrt(sum);			
		if (Double.isNaN(dist))
			dist = 0.0;
		
		return dist;
	}

}
