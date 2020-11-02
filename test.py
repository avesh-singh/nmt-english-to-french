from model import *
from util import *
import random
import pickle


def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        enc_hidden = encoder.init_hidden().to(device)
        input_tensor = sentence_to_tensor(input_lang, sentence).to(device)

        for e in range(input_tensor.size(0)):
            enc_output, enc_hidden = encoder(input_tensor[e], enc_hidden)

        dec_input = torch.tensor([[SOS_index]], dtype=torch.long, device=device)
        dec_hidden = enc_hidden
        output_sentence = []
        for d in range(MAX_TARGET_LENGTH):
            dec_output, dec_hidden = decoder(dec_input, dec_hidden)
            _, idx = dec_output.topk(1, -1)
            if idx.item() == EOS_index:
                break
            output_sentence.append(output_lang.index2word[idx.item()])
            dec_input = idx.squeeze().detach().to(device)
    return ' '.join(output_sentence)


def evaluate_random_sample(n=10):
    input_lang = pickle.load(open('data/input_language.pkl', 'rb'))
    output_lang = pickle.load(open('data/output_language.pkl', 'rb'))
    pairs = pickle.load(open('data/training_pairs.pkl', 'rb'))
    encoder = EncoderRNN(EMBEDDING_SIZE, input_lang.n_words, HIDDEN_SIZE)
    decoder = DecoderRNN(EMBEDDING_SIZE, output_lang.n_words, HIDDEN_SIZE, output_lang.n_words)
    encoder.load_state_dict(torch.load('model/enc_simple.pt'))
    decoder.load_state_dict(torch.load('model/dec_simple.pt'))
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        output_sentence = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        print('<', output_sentence)
        print('=', pair[1])
        print()


evaluate_random_sample()
