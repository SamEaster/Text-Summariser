from nltk.translate.bleu_score import sentence_bleu

weights = (1, 0, 0, 0)

def rouge_1(text: str, summary: str):
  text_s = set(text.split())
  summ_s = set(summary.split())
  cnt=0
  for t in summary.split():
    if t in text.split():
      cnt+=1

  precision = cnt/len(text_s)

  recall = cnt/len(summ_s)

  f1_score = 2*precision*recall/(precision+recall+1e-8)

  return {
      'precision': precision,
      'recall': recall,
      'f1_score': f1_score
    }

def calculate_score(x_tr, y_tr, model_output):
  p_score=0
  r_score=0
  f1_score=0
  bleu_score = 0
  
  sample = len(x_tr)//(128*2)
  for i in tqdm.tqdm(range(sample+1)):
      bleu_score += sentence_bleu(y_tr[i], pred, weights=weights)
      score_code = rouge_1(y_tr[i], model_output[i])
  
      p_score += score_code['precision']
      r_score += score_code['recall']
      f1_score += score_code['f1_score']
  
  
  print(f"Precision: {(precision/sample):.4f}  ---- {(p_score/sample):.4f}")
  print(f"Recall: {(recall/sample):.4f}  ---- {(r_score/sample):.4f}")
  print(f"Fmeasure: {(fmeasure/sample):.4f}  ---- {(f1_score/sample):.4f}")
  print(f"Bleu score {bleu_score/sample}")
