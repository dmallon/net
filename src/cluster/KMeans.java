package cluster;

import java.util.Random;

// The K-Means algorithm
public class KMeans {
	private int numClusters;
	private int numInputs;
	private char[] classes;
	private double[][] centers;
	
	private Random rnd = new Random();
	
	// Constructor to k-means algorithm class
	public KMeans(int numInputs, int numClusters, char[] classes){
		this.numClusters = numClusters;
		this.numInputs = numInputs;
		this.classes = classes;
		this.centers = new double[this.numClusters][this.numInputs];
	}
	
	// Train the cluster centers using the training set
	public void train(double[][] trainSet, int numSamples){
		double[][] newCenters = new double[this.numClusters][this.numInputs];
		double[] input = new double[this.numInputs];
		int[] cluster = new int[numSamples];
		int[] numMembers = new int[this.numClusters];
		int minIndex;
		double minDist;
		double dist;
		double sum;
		int index;
		
		// Initialize centers to random points chosen from the set
		for (int i = 0; i < this.numClusters; i++){
			index = rnd.nextInt(numSamples);		
			for(int j = 0; j < this.numInputs; j++){
				centers[i][j] = trainSet[j][index];
			}
		}
		// Loop var
		boolean loop = true;
		
		// Loop until the stopping condition is met
		do{
			// Initialize more arrays to 0
			for(int i = 0; i < this.numClusters; i++){
				numMembers[i] = 0;
			}
			
			for(int i = 0; i < this.numClusters; i++){
				for(int j = 0; j < this.numInputs; j++){
					newCenters[i][j] = 0.0;
				}
			}
			
			// Process each sample in the set
			for(int i = 0; i < numSamples; i++){
				// Get point i
				for (int j = 0; j < this.numInputs; j++){
					input[j] = trainSet[j][i];
				}
				
				// Find the center with the minimum distance to point i
				minIndex = 0;
				minDist = 1000000000.00;
				for (int j = 0; j < this.numClusters; j++){
					dist = distance(input, centers[j]);
					if( dist < minDist){
						minIndex = j;
						minDist = dist;
					}
				}
				cluster[i] = minIndex;
				numMembers[minIndex]++;
			}
			
			// Calculate the new centers based on member points
			for (int i = 0; i < numSamples; i++){
				for(int j = 0; j < this.numInputs; j++){
					newCenters[cluster[i]][j] += trainSet[j][i];
				}
			}			
			for (int i = 0; i < this.numClusters; i++){
				for (int j = 0; j < this.numInputs; j++){
					newCenters[i][j] = newCenters[i][j] / (double)numMembers[i];
				}
			}
			
			// Calculate sum of distance between old and new centers
			sum = 0.0;
			for (int i = 0; i < this.numClusters; i++){
				sum += distance(newCenters[i], centers[i]);
			}
			// If the Centers have changed by less than 0.0001, exit and finish
			if(sum < 0.0001)
				loop = false;
			
			// Copy the new centers into centers[]
			for (int i = 0; i < numClusters; i++){
				System.arraycopy(newCenters[i], 0, centers[i], 0, this.numInputs);
			}
		}while(loop == true);
	}
	
	// Process the test set on the trained clustering
	public void test(double[][] testSet, int numSamples, char[] classifier){
		double[] input = new double[this.numInputs];
		int[][] results = new int[this.numClusters][this.numClusters];
		int minIndex;
		double minDist;
		double dist;
		
		// Initialize the results matrix to 0
		for(int i = 0; i < this.numClusters; i++){
			for (int j = 0; j < this.numClusters; j++){
				results[i][j] = 0;
			}
		}
		
		// Process each sample in the set
		for(int i = 0; i < numSamples; i++){
			for (int j = 0; j < this.numInputs; j++){
				input[j] = testSet[j][i];
			}
			// Find cluster with minimum distance from center to point i
			minIndex = 0;
			minDist = 1000000000.00;
			for (int j = 0; j < this.numClusters; j++){
				dist = distance(input, centers[j]);
				if( dist < minDist){
					minIndex = j;
					minDist = dist;
				}
			}
			
			// Increment class counter in correct cluster
			for(int j = 0; j < this.numClusters; j++){
				if(classifier[i] == this.classes[j]){
					results[minIndex][j]++;
				}
			}
		}
		// Print out cluster structure
		for(int i = 0; i < this.numClusters; i++){
			System.out.print("Cluster " + (i + 1) + ": ");
			for (int j = 0; j < this.numClusters; j++){
				System.out.print(results[i][j] + " " + classes[j] + "'s  ");
			}
			System.out.println();
		}
		
		// Calculate the purity for the clustering
				int sum = 0;
				int max;
				for(int i = 0; i < this.numClusters; i++){
					max = 0;
					for (int j = 0; j < this.numClusters; j++){
						if(results[i][j] > max)
							max = results[i][j];
					}
					sum += max;
				}
				System.out.println("Cluster Purity: " + ((double)sum/(double)numSamples));
	}
	
	// Calculates Euclidean distance
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
