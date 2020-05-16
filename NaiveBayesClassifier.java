import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Your implementation of a naive bayes classifier. Please implement all four methods.
 */

public class NaiveBayesClassifier implements Classifier {
	int vocabSize;
	final int delta = 1;
	Map<Label, Integer> docsPerLabel;
	Map<Label, Integer> wordsPerLabel;
	//Map<String, Integer> cntEachWordPos = new HashMap<String, Integer>();;
	Map<String, Integer> cntEachWordPos; 
	Map<String, Integer> cntEachWordNeg ;
	//Map<String, Integer> cntEachWordNeg = new HashMap<String, Integer>();;
    /**
     * Trains the classifier with the provided training data and vocabulary size
     */
    @Override
    public void train(List<Instance> trainData, int v) {
       
        cntEachWordPos = new HashMap<String, Integer>();;
    	cntEachWordNeg = new HashMap<String, Integer>();;
    	this.vocabSize = v;
        // Hint: First, calculate the documents and words counts per label and store them.
    	docsPerLabel = getDocumentsCountPerLabel(trainData);
    	wordsPerLabel = getWordsCountPerLabel(trainData);
    	// Then, for all the words in the documents of each label, count the number of occurrences of each word.
    	//cntEachWordPos
    	//cntEachWordNeg = new HashMap<String, Integer>();
    	for(Instance curr: trainData){
    		List<String> currWords = curr.words;
    		if(curr.label == Label.POSITIVE){
	    		incWrdCnt(cntEachWordPos, currWords);
    		}	
    		else{
    			incWrdCnt(cntEachWordNeg, currWords);
    		}
    	}
    	    	
    	// Save these information as you will need them to calculate the log probabilities later.
        //
        // e.g.
        // Assume m_map is the map that stores the occurrences per word for positive documents
        // m_map.get("catch") should return the number of "catch" es, in the documents labeled positive
        // m_map.get("asdasd") would return null, when the word has not appeared before.
        // Use m_map.put(word,1) to put the first count in.
        // Use m_map.replace(word, count+1) to update the value
    }

    public void incWrdCnt (Map<String, Integer> map, List<String> wrds){
		for(String currWord: wrds){
			if(map.get(currWord) == null){
				map.put(currWord, 1);
			}
			else{
				map.replace(currWord, map.get(currWord) + 1);
			}
		}
		/*for(String currWord: currWords){
		if(cntEachWordPos.get(currWord) == null){
			cntEachWordPos.put(currWord, 1);
		}
		else{
			cntEachWordPos.replace(currWord, cntEachWordPos.get(currWord) + 1);
		}
	}*/
	}
    /*
     * Counts the number of words for each label
     */
    @Override
    public Map<Label, Integer> getWordsCountPerLabel(List<Instance> trainData) {
        
    	Map<Label,Integer> map = new HashMap<Label, Integer>();
    	int cntPos = 0;
    	int cntNeg = 0;
    	for(Instance curr: trainData){
    		if(curr.label == Label.POSITIVE){
    			cntPos = cntPos + curr.words.size();
    		}
    		else{
    			cntNeg = cntNeg + curr.words.size();
    		}
    	}
    	map.put(Label.POSITIVE, cntPos);
    	map.put(Label.NEGATIVE, cntNeg);
        return map;
    }


    /*
     * Counts the total number of documents for each label
     */
    @Override
    public Map<Label, Integer> getDocumentsCountPerLabel(List<Instance> trainData) {
        
    	Map<Label, Integer> map = new HashMap<Label, Integer>();
    	int numPos = 0;
    	int numNeg = 0;
    	for(Instance curr: trainData){
    		if ((curr.label) ==(Label.POSITIVE)){
    			numPos++;
    		}
    		else numNeg++;
    	}
    	map.put(Label.POSITIVE, numPos);
    	map.put(Label.NEGATIVE, numNeg);
        return map;
    }


    /**
     * Returns the prior probability of the label parameter, i.e. P(POSITIVE) or P(NEGATIVE)
     */
    private double p_l(Label label) {
        
        // Calculate the probability for the label. No smoothing here.
        // Just the number of label counts divided by the number of documents.
    	double pr_prob = (double)docsPerLabel.get(label)/(double)(docsPerLabel.get(Label.NEGATIVE) + docsPerLabel.get(Label.POSITIVE));
        return pr_prob;
    }

    /**
     * Returns the smoothed conditional probability of the word given the label, i.e. P(word|POSITIVE) or
     * P(word|NEGATIVE)
     */
    private double p_w_given_l(String word, Label label) {
        
        // Calculate the probability with Laplace smoothing for word in class(label)
    	//p_w_g_l = 
    	//get the cnt of w in l
    	int num = 0;
    	int den = wordsPerLabel.get(label) + (vocabSize* delta);
    	if(label == Label.POSITIVE){
    		if(cntEachWordPos.get(word) == null){
    			num = delta;
    		}
    		else{
    		num = cntEachWordPos.get(word) + delta;
    		}
    	}
    	else{
    		if(cntEachWordNeg.get(word) == null){
    			num = delta;
    		}
    		else{
    		num = cntEachWordNeg.get(word) + delta;
    		}
    	}
    	//get the sum of cnt of each word in vocab dict in label l
    	//System.out.println(num/den);
    	double ret = (double)num/(double)den;
        return ret;
    }

    /**
     * Classifies an array of words as either POSITIVE or NEGATIVE.
     */
    @Override
    public ClassifyResult classify(List<String> words) {
        // Sum up the log probabilities for each word in the input data, and the probability of the label
        // Set the label to the class with larger log probability
    	ClassifyResult finPred = new ClassifyResult();
    	List<Double> logProbw_g_pos = new ArrayList<Double>();
    	List<Double> logProbw_g_neg = new ArrayList<Double>();
    	for(String curr: words){
    		logProbw_g_pos.add(Math.log(p_w_given_l(curr, Label.POSITIVE)));
    		logProbw_g_neg.add(Math.log(p_w_given_l(curr, Label.NEGATIVE)));
    	}
    	double g_w_Pos = 0;
    	double g_w_Neg = 0;
    	for(int i = 0; i < words.size(); i++){
    		g_w_Neg += logProbw_g_neg.get(i);
    		g_w_Pos += logProbw_g_pos.get(i);
    	}
    	g_w_Neg += Math.log(p_l(Label.NEGATIVE));
    	g_w_Pos += Math.log(p_l(Label.POSITIVE));
    	if(g_w_Pos >= g_w_Neg){
    		finPred.label = Label.POSITIVE;
    	}
    	else{
    		finPred.label = Label.NEGATIVE;
    	}
    	finPred.logProbPerLabel = new HashMap<Label, Double>();
    	finPred.logProbPerLabel.put(Label.POSITIVE, g_w_Pos);
    	finPred.logProbPerLabel.put(Label.NEGATIVE, g_w_Neg);
        return finPred;
    }


}
