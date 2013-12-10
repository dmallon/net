package cluster;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class ACOC {

	int numInputs;
	int numClusters;
	int numSamples;
	Node[][] graph;
	Ant[] ants;
	double antRatio = 1;
	double beta = 0.5;
	double q = 0.5;
	double evapRate = 0.5;
	Random r = new Random();
	Ant globalBest = null;
	
	
	public ACOC(int numInputs, int numClusters, int numSamples){
		this.numInputs = numInputs;
		this.numClusters = numClusters;
		this.numSamples = numSamples;
		this.graph = new Node[numSamples][numClusters];
		this.ants = new Ant[(int) (numSamples * antRatio)];
		
		// Initialize pheromones to random small values
		for (int i = 0; i < graph.length; i++) {
			for (int j = 0; j < graph[i].length; j++) {
				graph[i][j] = new Node(r.nextDouble());
			}
		}
		
		// Initialize ants
		for (int i = 0; i < ants.length; i++) {
			ants[i] = new Ant(numInputs, numSamples, numClusters);
		}
		
	}
	
	public void train(double[][] trainSet, int epochs){
		
		// Invert training set in X matrix
		double[][] X = new double[numSamples][numInputs];
		for (int i = 0; i < X.length; i++) {
			for (int j = 0; j < X[i].length; j++) {
				X[i][j] = trainSet[j][i];
			}
		}
		
		// Set up input space vector to sample from
		ArrayList<Double> inputSpace = new ArrayList<>();
		for (int i = 0; i < trainSet.length; i++) {
			for (int j = 0; j < trainSet[i].length; j++) {
				inputSpace.add(trainSet[i][j]);
			}
		}
		
		// Initialize center matrix for each ant
		for (int i = 0; i < ants.length; i++) {
			for (int j = 0; j < ants[i].C.length; j++) {
				for (int k = 0; k < ants[i].C[j].length; k++) {
					ants[i].C[j][k] = inputSpace.get(r.nextInt(inputSpace.size()));
				}
			}
		}
		
		
		for (int e = 0; e < epochs; e++) {
			
			Ant localBest;
			
			// Process each ant
			for (int a = 0; a < ants.length; a++) {
				
				// Select unvisited data object for each ant
				ArrayList<Integer> unvisited = new ArrayList<>();
				for (int i = 0; i < ants[a].visited.length; i++) {
					if (ants[a].visited[i] == -1) {
						unvisited.add(i);
					}
				}
				int randomUnvisitedIndex = r.nextInt(unvisited.size());
				int dataObjectIndex = unvisited.get(randomUnvisitedIndex);
				ants[a].selectedObject = dataObjectIndex;
				
				// Select cluster
				int selectedCluster = -1;
				// Use exploitation
				if (r.nextDouble() < q) {
					
					int max = -1;
					for (int j = 0; j < numClusters; j++) {
						
						// Calculate distance
						double sum = 0.0;
						for (int n = 0; n < numInputs; n++) {
							sum += Math.pow(X[ants[a].selectedObject][n] - ants[a].C[j][n], 2);
						}
						double heuristic = Math.pow(1/Math.sqrt(sum), beta);
						double pheromone = graph[ants[a].selectedObject][j].pheromone;
						double product = heuristic * pheromone;
						if (product > max) {
							max = j;
						}
					}
					selectedCluster = max;
					
				}
				// Use exploration
				else{
					double[] clusterProb = new double[numClusters];
					
					for (int j = 0; j < numClusters; j++) {
						
						// Numerator
						// Calculate distance
						double sum = 0.0;
						for (int n = 0; n < numInputs; n++) {
							sum += Math.pow(X[ants[a].selectedObject][n] - ants[a].C[j][n], 2);
						}
						double heuristic = Math.pow(1/Math.sqrt(sum), beta);
						double pheromone = graph[ants[a].selectedObject][j].pheromone;
						double numerator = heuristic * pheromone;
						
						// Denominator
						double denominator = 0.0;
						for (int j2 = 0; j2 < numClusters; j2++) {
							double sum2 = 0.0;
							for (int n = 0; n < numInputs; n++) {
								sum2 += Math.pow(X[ants[a].selectedObject][n] - ants[a].C[j2][n], 2);
							}
							double heuristic2 = Math.pow(1/Math.sqrt(sum2), beta);
							double pheromone2 = graph[ants[a].selectedObject][j2].pheromone;
							denominator += heuristic2 * pheromone2;
						}
						
						clusterProb[j] = numerator/denominator;
					}
					
					// Choose cluster based on probability
					double p = Math.random();
					double cumulativeProbability = 0.0;
					for (int j = 0; j < clusterProb.length; j++) {
						
					    cumulativeProbability += clusterProb[j];
					    if (p <= cumulativeProbability) {
					        selectedCluster = j;
					    }
					}
				}
				
				/////////////////////
				// Update ant info //
				/////////////////////
				
				// Update ant memory
				ants[a].visited[ants[a].selectedObject] = selectedCluster;
				
				// Update ant weight matrix
				ants[a].W[ants[a].selectedObject][selectedCluster] = 1;
				
				// Update ant center matrix
				for (int j = 0; j < numClusters; j++) {
					for (int n = 0; n < numInputs; n++) {
						double numerator = 0.0;
						for (int m = 0; m < numSamples; m++) {
							numerator += ants[a].W[m][j] * X[m][n];
						}
						double denominator = 0.0;
						for (int m = 0; m < numSamples; m++) {
							denominator += ants[a].W[m][j];
						}
						
						ants[a].C[j][n] = numerator/denominator;
					}
				}
				
				// If all samples have been visited, break and calculate objective functions for ants
				if (!Arrays.asList(ants[a].visited).contains(-1)){
					break;
				}
				
			}
			
			// Calculate objective function values of each ant
			
			for (int a = 0; a < ants.length; a++) {
				
				double sum1 = 0.0;
				for (int i = 0; i < numSamples; i++) {
					double sum2 = 0.0;
					for (int j = 0; j < numClusters; j++) {
						
						double distSum = 0.0;
						for (int v = 0; v < numInputs; v++) {
							distSum += X[i][v] - ants[a].C[j][v];
						}
						sum2 += ants[a].W[i][j] * Math.sqrt(distSum);
					}
					sum1 += sum2;
				}
				
				ants[a].fitness = sum1;
			}
			
			// Sort ants by fitness value
			Arrays.sort(ants);
			localBest = ants[0];
			
			// If local best fitness value is greater than the global best fitness value, 
			// then global best = local best
			if (globalBest != null) {
				if (localBest.fitness > globalBest.fitness) {
					globalBest = localBest;
				}
			}
			else{
				globalBest = localBest;
			}
			
			// Update pheromone matrix
			for (int i = 0; i < graph.length; i++) {
				for (int j = 0; j < graph[i].length; j++) {
					graph[i][j].pheromone = (1 - evapRate) * graph[i][j].pheromone + 0.3 * localBest.fitness + 0.6 * globalBest.fitness;
				}
			}
			
		} // end epochs
	}
}
