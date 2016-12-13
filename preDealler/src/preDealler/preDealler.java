package preDealler;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations;  
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;  
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;  
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;  
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;  
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;  
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;  
import edu.stanford.nlp.ling.CoreLabel;  
import edu.stanford.nlp.pipeline.Annotation;  
import edu.stanford.nlp.pipeline.StanfordCoreNLP;  
import edu.stanford.nlp.util.CoreMap;  

public class preDealler {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Properties props = new Properties();  
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma");  
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        File file = new File("/home/wanghao/workspace/cl/wikiqa/sentences.txt");
        File wfile = new File("/home/wanghao/workspace/cl/wikiqa/lemma_sen.txt");
        BufferedReader reader = null;
        FileWriter writer = null;
        try {
			reader=new BufferedReader(new FileReader(file));
			writer = new FileWriter(wfile);
			String in = null, text = null;
			while ((in = reader.readLine()) != null){
				text = in.substring(in.indexOf('\t')+1);
				writer.write(in.substring(0, in.indexOf('\t'))+'\t');
				Annotation document = new Annotation(text); 
		        pipeline.annotate(document); 
		        List<CoreMap> sentences = document.get(SentencesAnnotation.class);
		        for(CoreMap sentence: sentences) {
		        	for (CoreLabel token: sentence.get(TokensAnnotation.class)){
		                String lema=token.get(LemmaAnnotation.class);  
		                writer.write(lema+' ');
		        	}
		        	writer.write("\n");
		        }
			}
			reader.close();
			writer.close();
	        
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
	}

}
