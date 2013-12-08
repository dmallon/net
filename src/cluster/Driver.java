package cluster;

import java.util.Scanner;
import java.io.File; 
import java.io.IOException;

// Main driver class for all network types
public class Driver{
	
//// Main Program Entry	////
	public static void main(String[] args) throws IOException, InterruptedException {
		// Initialize all parameters
		int numInputs, numSamples, algorithm, inFile;
		
		int epochs = 0;	
		
		int numClasses;
		double rate = 0.9;
		
		boolean classFirst;
		
		char[] trainExp;
		char[] testExp;
		char[] classes;
		
		double[][] trainSet;
		double[][] testSet;
		
		String fileName;
		
		Scanner keyscan;
		Scanner filescan;
		
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
		numSamples = keyscan.nextInt();
		
		switch(inFile){
			case 1:
				fileName = "data/wine.data";
				break;
			case 2:
				fileName = "data/voting-record.data";
				break;
			case 3:
				fileName = "data/transfusion.data";
				break;
			case 4:
				fileName = "data/tic-tac-toe.data";
				break;
			case 5:
				fileName = "data/mammogram.data";
				break;
			case 6:
				fileName = "data/liver.data";
				break;
			case 7:
				fileName = "data/iris.data";
				break;
			case 8:
				fileName = "data/fertility.data";
				break;
			case 9:
				fileName = "data/banknote-auth.data";
				break;
			case 10:
				fileName = "data/balance-scale.data";
				break;
			default:
				fileName = "data/wine.data";
		}

		file1 = new File(fileName);
		
		filescan = new Scanner(file1);
		filescan.useDelimiter(",|\\n|\\r\\n");
		
		// Read class position
		classFirst = filescan.nextBoolean();
		
		//Read number of classes from first line
		numClasses = filescan.nextInt();
	
		//Read value of n from second line
		numInputs = filescan.nextInt();
		
		trainSet = new double[numInputs][numSamples];
		testSet = new double[numInputs][numSamples];
		
		classes = new char[numClasses];
		
		trainExp = new char[numSamples];
		testExp = new char[numSamples];
				
		// Read in list of possible classes
		for (int i = 0; i < numClasses; i++){
			classes[i] = filescan.next().trim().charAt(0);
		}
		
		// Read in training set vectors
		for (int i = 0; i < numSamples; i++){
			if(classFirst)
				trainExp[i] = filescan.next().charAt(0);
			
			for (int j = 0; j < numInputs; j++){
				trainSet[j][i] = Double.parseDouble(filescan.next().trim());
			}
			
			if(!classFirst)
				trainExp[i] = filescan.next().charAt(0);
		}
		
		// Read in test set vectors
		for (int i = 0; i < numSamples; i++){
			if(classFirst)				
				testExp[i] = filescan.next().charAt(0);
			
			for (int j = 0; j < numInputs; j++){
				testSet[j][i] = Double.parseDouble(filescan.next().trim());
			}
			
			if(!classFirst)				
				testExp[i] = filescan.next().charAt(0);
		}			
		
		// Activate the requested algorithm to perform clustering
		switch(algorithm){
			case 1:
				CompNet net = new CompNet(numInputs, numClasses, rate, classes);
				net.train(trainSet, numSamples, epochs);
				net.test(testSet, numSamples, testExp);
				break;
			case 2:
				KMeans km = new KMeans(numInputs, numClasses, classes);
				km.train(trainSet, numSamples);
				km.process(testSet, numSamples, testExp);
				break;
			case 3:
				DBScan dbs = new DBScan(numInputs, classes, testSet, numSamples);
				dbs.cluster(testExp);
				break;
			case 4:
				break;
			case 5: 
				break;		
		}		
		
		filescan.close();
		keyscan.close();		
	}

}
