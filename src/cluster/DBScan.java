package cluster;

import java.util.ArrayList;

public class DBScan {
	
	private int minPts;
	private double theta;
	private int numInputs;
	private char[] classes;
	
	public DBScan(int numInputs, char[] classes){
		this.numInputs = numInputs;
		this.classes = classes;
	}
	
	public void cluster(double[][] trainSet, int numSamples, char[] classifier){
		
	}
	
	private void expandCluster(double[] point, ArrayList<Integer> neighbors, int cluster){
		
	}
	
	private ArrayList<Integer> nbhood(double[] point){
		return null;
	}
}
