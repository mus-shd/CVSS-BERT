import torch
import torch.nn.functional as F
from torch.autograd import Variable
import re



def connectivity_tensor_computation(classifier, input_ids, attention_mask, verbose=False):
    """
    Compute the connectivity tensor for a given input. The connectivity tensor contains the importance score of each input token obtained using gradient-based input saliency method.
    Args:
        classifier: HuggingFace Transformers model for classification.
        input_ids: BERT model input_ids
        attention_mask: BERT model attention_mask
    """
    input_embedding = classifier.get_input_embeddings()
    vocab_size = input_embedding.weight.shape[0]

    input_ids_one_hot = torch.nn.functional.one_hot(input_ids, num_classes=vocab_size)
    input_ids_one_hot = input_ids_one_hot.type(torch.float)
    input_ids_one_hot = Variable(input_ids_one_hot, requires_grad=True) #to allow the computation of the gradients with respect to the input 
    if verbose == True:
        print("input grad variable:", input_ids_one_hot.grad)

    #Calculate the input embeddings manually and pass them to the model through the inputs_embeds argument
    inputs_embeds = torch.matmul(input_ids_one_hot, input_embedding.weight)
    embedding_dim = input_embedding.weight.shape[1]
    inputs_embeds = torch.mul(inputs_embeds, torch.cat([attention_mask.unsqueeze(1)]*embedding_dim, dim=1))


    outputs = classifier(inputs_embeds=inputs_embeds.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))


    if verbose == True:
        print("output logits:", outputs.logits)

    predicted_label = torch.max(F.softmax(outputs.logits, dim=1), dim=1)[1].item()
    if verbose == True:
        print("predicted label (after softmax):", predicted_label)
        print("score for predicted label (after softmax):", torch.max(F.softmax(outputs.logits, dim=1), dim=1)[0].item())
    outputs.logits[0][predicted_label].backward() #compute the gradient of the logit (predicted, the one with the highest score)
    if verbose == True:
        print("input grad variable:", input_ids_one_hot.grad)                  #with respect to the input

    connectivity_tensor = torch.linalg.norm(input_ids_one_hot.grad, dim=1)
    connectivity_tensor = connectivity_tensor/torch.max(connectivity_tensor)
    return connectivity_tensor



def top_k_tokens(text, tokenizer, classifier, k=5):
    """
    Returns the list of input tokens (tokenized representation) along with the indices, the values and the connectivities (importance score) of the top k tokens.
    """
    text_encoding = tokenizer(text, truncation=True, padding=True, max_length=128)
    input_ids = torch.tensor(text_encoding['input_ids'])
    attention_mask = torch.tensor(text_encoding['attention_mask'])
    connectivity_tensor = connectivity_tensor_computation(classifier, input_ids, attention_mask)
    
    indices_sorted_by_connectivity = torch.argsort(connectivity_tensor, descending=True)
    input_tokens = tokenizer.convert_ids_to_tokens(list(input_ids))
    top_k_indices = indices_sorted_by_connectivity[:k]
    top_k_connectivity_weight = connectivity_tensor[top_k_indices]
    top_k_tokens = [input_tokens[i] for i in top_k_indices.tolist()]
    
    return {'input_tokens': input_tokens, 'top_k_tokens': top_k_tokens, 'top_k_indices': top_k_indices.tolist(), 'top_k_connectivity_weight': top_k_connectivity_weight.tolist()}





def print_texts_with_top_influential_words_in_bold(input_text_str, tokenizer, classifier, k=5):
    """
    print texts with the top_k relevant tokens (as determined using gradient-based input saliency method) in bold.
    """
    #input_text_str: python string coreesponding to the raw textual input
    #top_k: int representing the maximum number of top words to consider
    
    text_encoding = tokenizer(input_text_str, truncation=True, padding=True, max_length=128)
    input_ids = torch.tensor(text_encoding['input_ids'])
    attention_mask = torch.tensor(text_encoding['attention_mask'])
    
    connectivity_tensor = connectivity_tensor_computation(classifier, input_ids, attention_mask)
    input_tokens = tokenizer.convert_ids_to_tokens(list(input_ids))
    
    #input_tokens: python list corresponding to the tokenized representation of the input
    #connectivity_tensor: pytorch tensor containing the norm of the gradient of the logit with respect to each input token
    
    BOLD = '\033[1m'
    END = '\033[0m'
    
    output_str = input_text_str
    indices_sorted_by_connectivity = torch.argsort(connectivity_tensor, descending=True)
    top_indices_sorted = indices_sorted_by_connectivity[:k]
    
    for position, score in zip(top_indices_sorted,
                                     connectivity_tensor[top_indices_sorted]):
        
        if input_tokens[position.item()] in ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']:
            continue
        
        #find the indices of every tokens containing the selected word (or token)
        indices_all_matches = [i for i, x in enumerate(input_tokens) if re.sub('^##', '', input_tokens[position.item()]) in x]
        #keep only the position intended by the model (when multiple occurences of the same word).
        #For example, if selected words occurs 3 times in the description, and the algorithms is mostly influenced by
        #second occurrence, then return 1, 3rd occurence return 2, etc
        position_of_the_intended_match = [i for i, x in enumerate(indices_all_matches) if x == position.item()]
        
        test_sub = re.escape(re.sub('^##', '', input_tokens[position.item()]))
        res = [i.start() for i in re.finditer(test_sub, output_str, re.IGNORECASE)]
        idx = position_of_the_intended_match[0]
        output_str = output_str[:res[idx]] + BOLD + output_str[res[idx]:res[idx]+len(test_sub)] + END + output_str[res[idx]+len(test_sub):]
        
    print(output_str)
    return output_str
    