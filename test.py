from model import *
from util import *
import random
import pickle
import matplotlib.pyplot as plt


def evaluate(encoder, decoder, attention, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = sentence_to_tensor(input_lang, sentence).to(device)
        attention_matrix = torch.zeros(input_tensor.size(0) + 1, MAX_TARGET_LENGTH, device=device)
        encoder_states, enc_hidden = encoder(input_tensor)
        encoder_states = encoder_states.view(input_tensor.size(0), -1)
        dec_input = torch.tensor([[SOS_index]], dtype=torch.long, device=device)
        dec_hidden = enc_hidden
        output_sentence = []
        for d in range(MAX_TARGET_LENGTH):
            weights = attention(dec_hidden, encoder_states, device)
            attention_matrix[1:, d] = weights.view(1, -1)
            context = torch.mm(weights.T, encoder_states).unsqueeze(0)
            dec_output, dec_hidden = decoder(dec_input, context)
            _, idx = dec_output.topk(1, -1)
            if idx.item() == EOS_index:
                attention_matrix = attention_matrix[:, :d + 1]
                break
            output_sentence.append(output_lang.index2word[idx.item()])
            dec_input = idx.squeeze().detach().to(device)
    return ' '.join(output_sentence), attention_matrix.cpu().numpy()


def evaluate_random_sample(n=10):
    input_lang = pickle.load(open('data/input_language.pkl', 'rb'))
    output_lang = pickle.load(open('data/output_language.pkl', 'rb'))
    pairs = pickle.load(open('data/training_pairs.pkl', 'rb'))
    encoder = EncoderRNN(EMBEDDING_SIZE, input_lang.n_words, HIDDEN_SIZE)
    decoder = DecoderRNN(EMBEDDING_SIZE, output_lang.n_words, HIDDEN_SIZE, output_lang.n_words)
    attention = Attn('dot', HIDDEN_SIZE)
    encoder.load_state_dict(torch.load('model/enc_simple.pt'))
    decoder.load_state_dict(torch.load('model/dec_simple.pt'))
    attention.load_state_dict(torch.load('model/attention.pt'))
    encoder.to(device)
    decoder.to(device)
    attention.to(device)
    encoder.eval()
    decoder.eval()
    attention.eval()
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        output_sentence, weights = evaluate(encoder, decoder, attention, pair[0], input_lang, output_lang)
        print('<', output_sentence)
        print('=', pair[1])
        print()
        row = i // 5
        col = i % 5
        print(row + 1, col + 1)
        # ax = plt.subplot(row + 1, col + 1, 1)
        plt.matshow(weights)
        plt.xticks(range(len(pair[0].split())), pair[0].split())
        words = output_sentence.split()
        plt.yticks(range(len(words)), words)
    # plt.show()


evaluate_random_sample()
