import json
import urllib.parse
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import sequence
from corelib.constant import (dict_ixtoword_path, dict_wordtoix_path,
                              caption_generation_url, base_url)
from logger.logging import RekogntionLogger
import numpy as np
import requests

logger = RekogntionLogger(name="caption_generator_utils")


def predict_captions(image, sequence):
    """     Image Vectorzation
    Args:
            *   image: ndarray of dimension (2048,1)
            *   sequence an ndarray of indexes of generated words
    Workflow:
            *   A numpy array feature vector in taken as input
                inference input dimension requires dimension of (2048,1)
            *   Now the image is further processed to make it a
                json format which is compatible to TensorFlow Serving input.
            *   Then a http post request is made at localhost:8501.
                The post request contain data and headers.
            *   Incase of any exception, it return relevant error message.
            *   output from TensorFlow Serving is further processed using
                and a word is generated against that input and then the same
                model is called again internally until the generated caption is
                over or its length exceeds 51.
            *   A sting of number of words =1 is returned as output
    Returns:
            *   Generated word for a caption .
    """
    #logger.info(msg="predict_caption called")
    in1 = image.tolist()
    in2 = sequence.tolist()
    headers = {"content-type": "application/json"}
    data = json.dumps({"signature_name": "serving_default", "inputs": {'input_2': in1, 'input_3': in2}})
    try:
        headers = {"content-type": "application/json"}
        url = urllib.parse.urljoin(base_url, caption_generation_url)
        json_response = requests.post(url, data=data, headers=headers)

    except requests.exceptions.HTTPError as errh:
        logger.error(msg=errh)
        return {"Error": "An HTTP error occurred."}
    except requests.exceptions.ConnectionError as errc:
        logger.error(msg=errc)
        return {"Error": "A Connection error occurred."}
    except requests.exceptions.Timeout as errt:
        logger.error(msg=errt)
        return {"Error": "The request timed out."}
    except requests.exceptions.TooManyRedirects as errm:
        logger.error(msg=errm)
        return {"Error": "Bad URL"}
    except requests.exceptions.RequestException as err:
        logger.error(msg=err)
        return {"Error": "Caption Predicition Not Working"}
    except Exception as e:
        logger.error(msg=e)
        return {"Error": "Caption Predicition Not Working"}
    predictions = json.loads(json_response.text)
    # print("Predicitons are ",predictions)
    res = predictions['outputs']
    preds = np.array(res)
    return preds


def greedyCaptionSearch(photo):
    """     Caption Generation
    Args:
            *   image: A feature vector of size 2048,1)
    Workflow:
            *   The inputted imgage together with a sequence is fed to the function
                predict_caption
            *   The sequence variable keeps on getting updated with
                new predicted words from the predict_caption

            *   The predict_caption function is called multiple times to
                generate the whole caption
            *   A string is returned containing generated caption
    Returns:
            *   A string with generated caption
    """
    in_text = 'startseq'
    a_file = open(dict_ixtoword_path, "rb")
    ixtoword = pickle.load(a_file)
    a_file.close()
    b_file = open(dict_wordtoix_path, "rb")
    wordtoix = pickle.load(b_file)
    b_file.close()
    max_length = 51
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        preds = predict_captions(photo, sequence)
        yhat = np.argmax(preds)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break

    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


def beam_search_predictions(image, beam_index=3):
    """     Caption Generation
    Args:
            *   image: A feature vector of size 2048,1)
            *   beam_index :beam_index to average and search for words
                in the given range
    Workflow:
            *   The inputted imgage together with the par_caps is fed to the function
                predict_caption
            *   The par_caps variable keeps on getting updated with
                new predicted words from the predict_caption

            *   The predict_caption function is called multiple times to
                generate the whole caption
            *   A string is returned containing generated caption
    Returns:
            *   A string with generated caption
    """

    a_file = open(dict_ixtoword_path, "rb")
    ixtoword = pickle.load(a_file)
    a_file.close()
    b_file = open(dict_wordtoix_path, "rb")
    wordtoix = pickle.load(b_file)
    b_file.close()
    max_length = 51
    start = [wordtoix["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = predict_captions(image, par_caps)
            # preds = model.predict([image,par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            # Getting the top <beam_index>(n) predictions and creating a
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])

        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]
    final_caption = []

    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption
