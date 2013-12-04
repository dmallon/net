package cluster;

import java.util.Scanner;
import java.io.File; 
import java.io.IOException;

// Main driver class for all network types
public class Driver{
	
//// Main Program Entry	////
	public static void main(String[] args) throws IOException, InterruptedException {
		// Initialize all parameters
		int inputs, samples, correct, error, fail, algorithm, inFile;
		
		int epochs = 0;	
		int layers = 1;
		
		int numClasses = 3;
		double rate = 0.001;
		
		boolean classFirst = true;
		
		int outputs = numClasses;
		
		char[] expected1;
		char[] expected2;
		char[] classes;
		
		double[][] trainSet;
		double[][] testSet;
		
		String fileName1;
		
		Scanner keyscan;
		Scanner filescan1;
		
		File file1;
		
		keyscan = new Scanner(System.in);
		
	// Take user input to set layers, epochs, samples, n, and training algorithm dynamically
		System.out.println("Select clustering algorithm: ");
		System.out.println("1. Competitive Neural Net");
		System.out.println("2. K-Means");
		System.out.println("3. DB-Scan");
		System.out.println("4. ASO");
		System.out.println("5. PSO");
		algorithm = keyscan.nextInt();
		
		if(algorithm == 1){
			System.out.println("Enter the number of hidden layers: ");
			layers = keyscan.nextInt();		
			
			System.out.println("Enter number of training epochs: ");
			epochs = keyscan.nextInt();
		}
		
		System.out.println("Select input file: ");
		System.out.println("1. wine.data");
		System.out.println("2. voting-record.data");
		System.out.println("3. transfusion.data");
		System.out.println("4. tic-tac-toe.data");
		System.out.println("5. mammogram.data");
		System.out.println("6. liver.data");
		System.out.println("7. iris.data");
		System.out.println("8. fertility.data");
		System.out.println("9. banknote-auth.data");
		System.out.println("10. balance-scale.data");
		inFile = keyscan.nextInt();
		
		
		System.out.println("Enter number of samples: ");
		samples = keyscan.nextInt();
		
		System.out.println("Enter a value for n: ");
		inputs = keyscan.nextInt();
		
		switch(inFile){
			case 1:
				fileName1 = "data/wine.data";
				break;
			case 2:
				fileName1 = "data/voting-record.data";
				break;
			case 3:
				fileName1 = "data/transfusion.data";
				break;
			case 4:
				fileName1 = "data/tic-tac-toe.data";
				break;
			case 5:
				fileName1 = "data/mammogram.data";
				break;
			case 6:
				fileName1 = "data/liver.data";
				break;
			case 7:
				fileName1 = "data/iris.data";
				break;
			case 8:
				fileName1 = "data/fertility.data";
				break;
			case 9:
				fileName1 = "data/banknote-auth.data";
				break;
			case 10:
				fileName1 = "data/balance-scale.data";
				break;
			default:
				fileName1 = "data/wine.data";
		}

		file1 = new File(fileName1);
		
		filescan1 = new Scanner(file1);
		filescan1.useDelimiter(",|\\n|\\r\\n");
		
		trainSet = new double[inputs][samples];
		testSet = new double[inputs][samples];
		
		classes = new char[numClasses];
		
		expected1 = new char[samples];
		expected2 = new char[samples];
				
		// Read in list of possible classes
		for (int i = 0; i < numClasses; i++){
			classes[i] = filescan1.next().trim().charAt(0);
		}
		
		// Read in training set vectors
		for (int i = 0; i < samples; i++){
			if(classFirst)
				expected1[i] = filescan1.next().charAt(0);
			
			for (int j = 0; j < inputs; j++){
				trainSet[j][i] = Double.parseDouble(filescan1.next().trim());
			}
			
			if(!classFirst)
				expected1[i] = filescan1.next().charAt(0);
		}
		
		// Read in test set vectors
		for (int i = 0; i < samples; i++){
			if(classFirst)				
				expected2[i] = filescan1.next().charAt(0);
			
			for (int j = 0; j < inputs; j++){
				testSet[j][i] = Double.parseDouble(filescan1.next().trim());
			}
			
			if(!classFirst)				
				expected2[i] = filescan1.next().charAt(0);
		}			
		
		// Activate the requested algorithm to perform clustering
		switch(algorithm){
			case 1:
				CompNet net = new CompNet(inputs, layers, outputs, rate, classes);
				net.train(trainSet, samples, epochs, classes);
				break;
			case 2:
				break;
			case 3:
				break;
			case 4:
				break;
			case 5: 
				break;		
		}		
		
		
		// Begin testing phase ///////////////////////////////////////
		correct = error = fail = 0;		
		
		// Check each test vector against each class network		

		
		// Output results and final error percentage
		System.out.println("Correct: " + correct + ", Error: " + error + ", Failed to Class: " + fail);
		System.out.println("Percent incorrect: " + (((double)(error + fail))/((double)(error + fail + correct)))*100.00 + "%");
		
		filescan1.close();
		keyscan.close();		
	}

}
