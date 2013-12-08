package cluster;

import java.util.ArrayList;

public class DBScan {
	
	private int minPts = 50;
	private double theta = 500.0;
	private int numInputs;
	private char[] classes;
	private boolean[] visited;
	private int numSamples;
	private double[][] set;
	
	int[] clusterLbl;
	
	// Constructor to setup the DBScan class
	public DBScan(int numInputs, char[] classes, double[][] set, int numSamples){
		this.numInputs = numInputs;
		this.classes = classes;
		this.set = set;
		this.numSamples = numSamples;
	}
	
	public void cluster(char[] classifier){
		int c = 0;		
		double[] point = new double[this.numInputs];
		boolean[] noise = new boolean[numSamples];
		ArrayList<Integer> nbhood;
		ArrayList<Integer[]> results = new ArrayList<Integer[]>();
		int noisePts[] = new int[this.classes.length];
		
		this.clusterLbl = new int[numSamples];
		this.visited = new boolean[numSamples];
		
		for (int i = 0; i < this.classes.length; i++){
			noisePts[i] = 0;
		}
		
		// Initialize visited matrix to all false
		for (int i = 0; i < numSamples; i++){
			this.visited[i] = false;
			noise[i] = false;
			this.clusterLbl[i] = -1;
		}
		
		for (int i = 0; i < numSamples; i++){
			if(!visited[i]){
				// Get point i
				for (int j = 0; j < this.numInputs; j++){
					point[j] = set[j][i];
				}
				
				// Mark the point as visited
				visited[i] = true;
				
				// Get the neighborhood for point i
				nbhood = this.getNbhood(point);
				if(nbhood.size() < this.minPts){
					noise[i] = true;
				}
				else{
					c++;
					expandCluster(point, nbhood, c, i);
				}
			}
		}
		
		for (int i = 0; i < this.numSamples; i++){
			if(this.clusterLbl[i] != -1){
				if(results.size() < this.clusterLbl[i] || results.isEmpty()){
					results.add(this.clusterLbl[i] - 1, new Integer[this.classes.length]);
					for (int j = 0; j < this.classes.length; j++){
						results.get(clusterLbl[i] - 1)[j] = 0;
					}
				}
				for (int j = 0; j < this.classes.length; j++){
					if(classifier[i] == this.classes[j])
						results.get(clusterLbl[i] - 1)[j]++;
				}
			}
			else{
				for (int j = 0; j < this.classes.length; j++){
					if(classifier[i] == this.classes[j])
						noisePts[j]++;
				}
			}			
		}	
		
		for (int i = 0; i < results.size(); i++){
			System.out.print("Cluster " + (i + 1) + ": ");
			for (int j = 0; j < this.classes.length; j++){
				System.out.print(results.get(i)[j] + " " + this.classes[j] + "'s  ");
			}
			System.out.println();
		}
		System.out.print("Noise Points: ");
		for (int j = 0; j < this.classes.length; j++){
			System.out.print(noisePts[j] + " " + this.classes[j] + "'s  ");
		}
		System.out.println();
		
		// Calculate the purity for the clustering
		int sum = 0;
		int max;
		for(int i = 0; i < results.size(); i++){
			max = 0;
			for (int j = 0; j < this.classes.length; j++){
				if(results.get(i)[j] > max)
					max = results.get(i)[j];
			}
			sum += max;
		}
		System.out.println("Cluster Purity: " + ((double)sum/(double)numSamples));
	}
	
	private void expandCluster(double[] point, ArrayList<Integer> neighbors, int c, int i){
		double[] pPrime = new double[this.numInputs];
		ArrayList<Integer> nbhPrime = new ArrayList<Integer>();
		
		this.clusterLbl[i] = c;
		
		for (int j = 0; j < neighbors.size(); j++){
			// Get the next point pPrime from the neighbors of point i
			for (int k = 0; k < this.numInputs; k++){
				pPrime[k] = set[k][neighbors.get(j)];
			}
			
			if(!visited[neighbors.get(j)]){
				visited[neighbors.get(j)] = true;
				nbhPrime = getNbhood(pPrime);
				if(nbhPrime.size() >= this.minPts){
					neighbors.addAll(nbhPrime);
				}
			}
			if(this.clusterLbl[neighbors.get(j)] == -1){
				this.clusterLbl[neighbors.get(j)] = c;
			}
		}
	}
	
	private ArrayList<Integer> getNbhood(double[] point){
		ArrayList<Integer> nbhood = new ArrayList<Integer>();
		double[] pPrime = new double[this.numInputs];
		
		for(int i = 0; i < this.numSamples; i++){
			for (int j = 0; j < this.numInputs; j++){
				pPrime[j] = set[j][i];
			}
			if(distance(point, pPrime) < this.theta){
				nbhood.add(i);
			}
		}
		return nbhood;
	}
	
	private double distance(double[] x1, double[] x2){
		double sum = 0.0;
		double dist;
		for(int i = 0; i < this.numInputs; i++){
			sum += Math.pow((x1[i] - x2[i]), 2.0);
		}
		dist = Math.sqrt(sum);
		
		if (Double.isNaN(dist))
			return 0.0;
		else
			return dist;
	}
}
