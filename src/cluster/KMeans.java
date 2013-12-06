package cluster;

import java.util.Random;

public class KMeans {
	private int numClusters;
	private int numInputs;
	private char[] classes;
	private double[][] centers;
	
	private Random rnd = new Random();
	
	public KMeans(int numInputs, int numClusters, char[] classes){
		this.numClusters = numClusters;
		this.numInputs = numInputs;
		this.classes = classes;
		this.centers = new double[this.numClusters][this.numInputs];
	}
	
	public void train(double[][] trainSet, int numSamples, char[] classifier){
		double[][] newCenters = new double[this.numClusters][this.numInputs];
		int[][] results = new int[this.numClusters][this.numClusters];
		double[] input = new double[this.numInputs];
		int[] cluster = new int[numSamples];
		int[] numMembers = new int[this.numClusters];
		int minIndex;
		double minDist;
		double dist;
		double sum;
		int index;
		
		for (int i = 0; i < this.numClusters; i++){
			index = rnd.nextInt(numSamples);		
			for(int j = 0; j < this.numInputs; j++){
				centers[i][j] = trainSet[j][index];
			}
		}
		boolean loop = true;
		
		do{
			for(int i = 0; i < this.numClusters; i++){
				numMembers[i] = 0;
			}
			
			for(int i = 0; i < this.numClusters; i++){
				for(int j = 0; j < this.numInputs; j++){
					newCenters[i][j] = 0.0;
				}
			}
			
			for(int i = 0; i < numSamples; i++){
				for (int j = 0; j < this.numInputs; j++){
					input[j] = trainSet[j][i];
				}
				
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
			
			sum = 0.0;
			for (int i = 0; i < this.numClusters; i++){
				sum += distance(newCenters[i], centers[i]);
			}
			if(sum < 0.0001)
				loop = false;
			
			for (int i = 0; i < numClusters; i++){
				System.arraycopy(newCenters[i], 0, centers[i], 0, this.numInputs);
			}
		}while(loop == true);
	}
	
	public void process(double[][] testSet, int numSamples, char[] classifier){
		double[] input = new double[this.numInputs];
		int[][] results = new int[this.numClusters][this.numClusters];
		int minIndex;
		double minDist;
		double dist;
		
		for(int i = 0; i < this.numClusters; i++){
			for (int j = 0; j < this.numClusters; j++){
				results[i][j] = 0;
			}
		}
		
		for(int i = 0; i < numSamples; i++){
			for (int j = 0; j < this.numInputs; j++){
				input[j] = testSet[j][i];
			}
			
			minIndex = 0;
			minDist = 1000000000.00;
			for (int j = 0; j < this.numClusters; j++){
				dist = distance(input, centers[j]);
				if( dist < minDist){
					minIndex = j;
					minDist = dist;
				}
			}

			for(int j = 0; j < this.numClusters; j++){
				if(classifier[i] == this.classes[j]){
					results[minIndex][j]++;
				}
			}
		}
		for(int i = 0; i < this.numClusters; i++){
			System.out.print("Cluster " + (i + 1) + ": ");
			for (int j = 0; j < this.numClusters; j++){
				System.out.print(results[i][j] + " " + classes[j] + "'s  ");
			}
			System.out.println();
		}
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
