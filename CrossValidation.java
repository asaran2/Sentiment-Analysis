import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.sun.org.apache.bcel.internal.generic.ClassGen;

public class CrossValidation {
    /*
     * Returns the k-fold cross validation score of classifier clf on training data.
     */
    public static double kFoldScore(Classifier clf, List<Instance> trainData, int k, int v) {
       
    	List<List<Instance>> subsets = new ArrayList<List<Instance>>();	//k folds
    	List<Instance> trainSubSet = new ArrayList<Instance>();	
    	//List<Instance> testSubSet = new ArrayList<Instance>();	//subset used to test
    	int sizeSubSet = trainData.size() / k;	//num instances in each fold
    	//int curr = 0;
    	
    	
    	for(int i = 0; i < k;i++){
    		//initialize k Instance lists for subset
    		subsets.add(new ArrayList<Instance>());
    	}
    	
    	//split the input data into k folds
    	int i = 0;
    	while(i <trainData.size()){	

    		subsets.get(i/sizeSubSet).add(trainData.get(i));
    		i++;
    	}

    	
		List<Double> acc = new ArrayList<Double>();	//holds accuracy for each fold
		
    	for( i = 0; i < k; i++){	//for each fold
    		trainSubSet = new ArrayList<Instance>();
    		clf = new NaiveBayesClassifier();
    		List<Instance> test = subsets.get(i);
    		//clf.train(input, v);
    		for(int j = 0; j < subsets.size();j++){
    			if(j != i){
    				for(int l = 0; l < subsets.get(j).size(); l++){
    					trainSubSet.add(subsets.get(j).get(l));
    				}
    				
    			}    			
    		}
    		clf.train(trainSubSet, v);
    		int correct = 0;
			for(Instance currInst: test){
				
				ClassifyResult result = clf.classify(currInst.words);
				if(currInst.label == result.label){
					correct++;
				}
			}
			acc.add((double)correct/(double)test.size());
	
		
    	}
    	
    	double score = 0;
    	for(int num = 0;  num < acc.size();num++){
    		//sSystem.out.println(acc.get(num));
    		score+=acc.get(num);
    	}
    	
    	
        return (score/(double)k);
    }
}

