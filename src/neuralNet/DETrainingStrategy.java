package neuralNet;

import java.util.Random;

public class DETrainingStrategy implements ITrainingStrategy {
	private int pop_size = 20;
	private int chrom_len;
	private double[][] pop;
	
	private double mut_rate;
	private double cross_rate;
	
	private Random rnd = new Random();

	@Override
	public void train(Network net, double[][] trainSet, int numSamples,
			int epochs, char[] classes) 
	{
		
		int t = 0;
		
		
		
		
	}
	
	private void mutate(double[] chrom){
		
	}
	
	private void crossover(){
		
	}

}
