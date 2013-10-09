package NerualNet;

import java.util.Scanner;
import java.io.File; 
import java.io.IOException;

public class NeuralNet {

	public static void main(String[] args) throws IOException {
		int type, inputs,samples, epochs;
		
		int outputs = 1;
		int layers = 0;
		int centers = 0;
		
		double rate;
		
		double[][] set1;
		double[][] set2;
		
		String fileName1;
		String fileName2;
		
		Scanner keyscan;
		Scanner filescan1;
		Scanner filescan2;
		
		Network net;
		
		File file1;
		File file2;
		
		rate = 0.01;
		
		keyscan = new Scanner(System.in);
		
		
		System.out.println("Select network type: ");
		System.out.println("1. MLP");
		System.out.println("2. RBF");
		type = keyscan.nextInt();
		
		if(type == 1){
			System.out.println("Enter the number of hidden layers: ");
			layers = keyscan.nextInt();
		}
		else{
			System.out.println("Enter the number of centers: ");
			centers = keyscan.nextInt();
		}
		
		System.out.println("Enter number of training epochs: ");
		epochs = keyscan.nextInt();
		
		System.out.println("Enter number of samples: ");
		samples = keyscan.nextInt();
		
		System.out.println("Enter a value for n: ");
		inputs = keyscan.nextInt();
		
		for (int a = 0; a < 4; a++){
			System.out.println("\n**************************************************************************");
			System.out.println("Train/Validate/Test Round " + (a + 1));
			fileName1 = "data/n" + inputs + "/" + a;
			System.out.println("Using file " + fileName1 + " for training");
			
			file1 = new File(fileName1);
			
			filescan1 = new Scanner(file1);
			
			set1 = new double[inputs+outputs][samples];
			
			for (int i = 0; i < samples; i++){
				for (int j = 0; j < inputs+outputs; j++){
					set1[j][i] = filescan1.nextDouble();
				}
			}
			
			fileName2 = "data/n" + inputs + "/" + ((a + 1)%4);
			System.out.println("Using file " + fileName2 + " for validation");
			
			file2 = new File(fileName2);
			
			filescan2 = new Scanner(file2);
			
			set2 = new double[inputs+outputs][samples];
	
			for (int i = 0; i < samples; i++){
				for (int j = 0; j < inputs+outputs; j++){
					set2[j][i] = filescan2.nextDouble();
				}
			}
			
			if(type == 1){
				net = new MLPNet(inputs, layers, outputs, rate);
			}
			else{
				net = new RBFNet(inputs, centers, outputs, rate, samples, set1);
			}	
			
			for(int i = 0; i < 20; i++){
				net.train(set1, samples, (epochs/20));				
				System.out.println("Average validation error: " + String.valueOf(net.process(set2, samples)) + "%");
			}
			
			filescan1.close();
			filescan2.close();
			
		
			fileName1 = "data/n" + inputs + "/" + ((a + 2)%4);
			System.out.println("\nUsing file " + fileName1 + " for testing");
			
			file1 = new File(fileName1);
			
			filescan1 = new Scanner(file1);
	
			for (int i = 0; i < samples; i++){
				for (int j = 0; j < inputs+outputs; j++){
					set1[j][i] = filescan1.nextDouble();
				}
			}
			
			System.out.println("Average test error: " + String.valueOf(net.process(set1, samples)) + "%");
			
			fileName1 = "data/n" + inputs + "/" + ((a + 3)%4);
			System.out.println("\nLeaving out file " + fileName1 + " for this round");
						
			filescan1.close();
		}
		keyscan.close();		
	}

}
