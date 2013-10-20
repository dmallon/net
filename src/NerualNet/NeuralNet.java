package NerualNet;

import java.util.Scanner;
import java.io.File; 
import java.io.IOException;

// Main driver class for all network types
public class NeuralNet implements Runnable{
	
/*** Setup requirements for multithreading the training process **************/	
	Network net;
	double[][] trainSet;
	int numSamples;
	int epochs;
	char[] classes;
	
	public NeuralNet(Network net, double[][] trainSet, int numSamples, int epochs, char[] classes){
		this.net = net;
		this.trainSet = trainSet;
		this.numSamples = numSamples;
		this.epochs = epochs;
		this.classes = classes;
	}
	public void run(){
		this.net.train(this.trainSet, this.numSamples, this.epochs, this.classes);
	}
/*************** End threading setup *******************************************/
	
//// Main Program Entry	
	public static void main(String[] args) throws IOException, InterruptedException {
		int type, inputs,samples, epochs;
		
		int outputs = 26;
		int layers = 0;
		int centers = 0;
		
	/** Set number of threads here **/
		int numThreads = 8;
	/********************************/
		
		char[] expected;
		char[] classes;
		
		double rate;
		
		double[][] set1;
		
		String fileName1;
		
		Scanner keyscan;
		Scanner filescan1;
		
		Thread[] threads;
		
		Network[] net;
		
		File file1;
		
		rate = 0.01;
		
		keyscan = new Scanner(System.in);
		
		
		System.out.println("Select network type: ");
		System.out.println("1. MLP");
		System.out.println("2. RBF");
		System.out.println("3. ANFIS");
		
		type = keyscan.nextInt();
		
		if(type == 1){
			System.out.println("Enter the number of hidden layers: ");
			layers = keyscan.nextInt();
		}
		else if (type == 2){
			System.out.println("Enter the number of centers: ");
			centers = keyscan.nextInt();
		}
		else if (type == 3){
			System.out.println("Test ANFIS: ");
		}
		
		System.out.println("Enter number of training epochs: ");
		epochs = keyscan.nextInt();
		
		System.out.println("Enter number of samples: ");
		samples = keyscan.nextInt();
		
		System.out.println("Enter a value for n: ");
		inputs = keyscan.nextInt();
		
		//System.out.println("Enter input file name: ");
		//fileName1 = keyscan.next();
		
		//////// Hardcode filename for now
		fileName1 = "data/letter-recognition.data";
		////////
		
		file1 = new File(fileName1);
		
		filescan1 = new Scanner(file1);
		filescan1.useDelimiter(",|\\n");
		
		set1 = new double[inputs][samples];
		
		classes = new char[outputs];
		
		expected = new char[samples];
		
		threads = new Thread[numThreads];
				
		// Read in list of possible classes
		for (int i = 0; i < outputs; i++){
			classes[i] = filescan1.next().charAt(0);
		}
		
		// Read in set vectors
		for (int i = 0; i < samples; i++){
			expected[i] = filescan1.next().charAt(0);
			
			for (int j = 0; j < inputs; j++){
				set1[j][i] = Integer.parseInt(filescan1.next());
			}
		}
		
		net = new Network[outputs];
		
		int t = 0;
		
		for (int i = 0; i < outputs; i++){
			if(type == 1)
				net[i] = new MLPNet(inputs, layers, 1, rate, classes[i]);
			else if (type == 2)
				net[i] = new RBFNet(inputs, centers, 1, rate, classes[i], samples, set1);
			
			threads[t] = new Thread(new NeuralNet(net[i], set1, samples, epochs, expected));
			threads[t].start();
			
			if(t == numThreads - 1){
				for (Thread thread : threads) {
					  thread.join();
				}
				t = 0;				
			}
			else
				t++;
		}
		
		for (Thread thread : threads) {
			  thread.join();
		}
		
		
		char out;
		for (int a = 0; a < 100; a++){
			int responses = outputs;
			for (int i = 0; i < outputs; i++){
				out = net[i].process(set1, a, expected[a]);
				if(out != '!'){
					if(out == expected[a])
						System.out.print(out + " ");
					else						
						System.out.print("Error(" + out + ") ");
				}
				else
					responses--;
			}
			if(responses == 0)
				System.out.print("Failed to classify");
			System.out.println();
		}
		
		filescan1.close();
		keyscan.close();		
	}

}
