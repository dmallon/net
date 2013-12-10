package cluster;

public class Ant implements Comparable<Ant>{
	double[][] C;
	int[][] W;
	int selectedObject;
	int[] visited;
	double fitness;
	
	public Ant(int numInputs, int numSamples, int numClusters){
		this.visited = new int[numSamples];
		this.C = new double[numClusters][numInputs];
		this.W = new int[numSamples][numClusters];
		
		// Initialize unvisited data objects with -1, the index will be used when visited.
		for (int i = 0; i < visited.length; i++) {
			visited[i] = -1;
		}
		
		// Initialize weight matrix
		for (int i = 0; i < W.length; i++) {
			for (int j = 0; j < W[i].length; j++) {
				W[i][j] = 0;
			}
		}
	}

	@Override
	public int compareTo(Ant ant) {
		return Double.compare(this.fitness, ant.fitness);
	}
}
