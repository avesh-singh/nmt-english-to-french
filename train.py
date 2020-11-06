from model import *
from util import *
from data import *
import pickle
import random

# Setup the model and train
input_lang, output_lang, train_pairs, valid_pairs, test_pairs = prepare_data('eng', 'fra')
encoder = EncoderRNN(EMBEDDING_SIZE, input_lang.n_words, HIDDEN_SIZE)
decoder = DecoderRNN(EMBEDDING_SIZE, output_lang.n_words, HIDDEN_SIZE, output_lang.n_words)
attn = Attn('dot', HIDDEN_SIZE)
hidden_state_mapper = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
enc_optimizer = torch.optim.Adam(encoder.parameters())
dec_optimizer = torch.optim.Adam(decoder.parameters())
if attn.method != 'dot':
    attn_optimizer = torch.optim.Adam(attn.parameters())
hs_mapper_optimizer = torch.optim.Adam(hidden_state_mapper.parameters())
criterion = torch.nn.NLLLoss()
teacher_forcing_ratio = 0.4
# N_ITER = 50000
EPOCHS = 5


def train(encoder, decoder, attention, input_tensor, target_tensor, record_outputs):
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()
    hidden_state_mapper.zero_grad()
    if attention.method != 'dot':
        attn_optimizer.zero_grad()
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)

    encoder_states, enc_hidden = encoder(input_tensor)
    encoder_states = encoder_states.view(input_tensor.size(0), -1)
    dec_hidden = hidden_state_mapper(enc_hidden)

    loss = torch.zeros(1, device=device)
    decoder_words = []
    decoder_targets = []
    if random.random() > teacher_forcing_ratio:
        decoder_input = torch.tensor([[SOS_index]], dtype=torch.long, device=device)
        for d in range(target_tensor.size(0)):
            energies = attention(dec_hidden[0], encoder_states, device)
            context = torch.mm(energies.T, encoder_states).unsqueeze(0)
            dec_out, dec_hidden = decoder(decoder_input, context)
            loss += criterion(dec_out[0], target_tensor[d])
            _, idx = dec_out.topk(1, -1)
            decoder_input = idx.squeeze().detach()
            if record_outputs:
                decoder_words.append(output_lang.index2word[idx.item()])
                decoder_targets.append(output_lang.index2word[target_tensor[d].item()])
            # if decoder_input.item() == EOS_index:
            #     break
        if record_outputs:
            print("no teacher forcing")
            print(">", ' '.join(decoder_words))
            print("=", ' '.join(decoder_targets))
    else:
        decoder_input = get_decoder_input(target_tensor).to(device)
        for d in range(decoder_input.size(0)):
            energies = attention(dec_hidden, encoder_states, device)
            context = torch.mm(energies.T, encoder_states).unsqueeze(0)
            dec_out, dec_hidden = decoder(decoder_input[d], context)
            loss += criterion(dec_out[0], target_tensor[d])
            if record_outputs:
                decoder_words.append(output_lang.index2word[dec_out.topk(1, -1)[1].item()])
                decoder_targets.append(output_lang.index2word[target_tensor[d].item()])
        if record_outputs:
            print("teacher forcing")
            print(">", ' '.join(decoder_words))
            print("=", ' '.join(decoder_targets))
    loss.backward()
    enc_optimizer.step()
    dec_optimizer.step()
    hs_mapper_optimizer.step()
    if attention.method != 'dot':
        attn_optimizer.step()
    return loss.item() / target_tensor.size(0)


def train_iters(encoder, decoder, attention):
    start = time.time()
    plot_loss = []
    valid_loss = []
    print_loss_total = 0
    plot_loss_total = 0
    print_every = 1500
    plot_every = 100
    # record_outputs = False
    for ep in range(EPOCHS):
        for i in range(len(train_pairs)):
            record_outputs = (i % print_every == 0)
            training_pair = pair_to_tensor(train_pairs[i], input_lang, output_lang)
            input_tensor = training_pair[0].to(device)
            target_tensor = training_pair[1].to(device)

            loss = train(encoder, decoder, attention, input_tensor, target_tensor, False)
            print_loss_total += loss
            plot_loss_total += loss

            if record_outputs:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print(f'{time_since(start, (i+1) / len(train_pairs))} ({i} {round(i/ len(train_pairs) * 100, 4)}%'
                      f' {round(print_loss_avg, 4)})')

            if i % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_loss_total = 0
                plot_loss.append(plot_loss_avg)
        valid_loss.append(validate(encoder, decoder, attention))
    return plot_loss, valid_loss


def validate(encoder, decoder, attention):
    with torch.no_grad():
        losses = []
        for i in range(len(valid_pairs)):
            tensor_pair = pair_to_tensor(valid_pairs[i], input_lang, output_lang)
            input_tensor = tensor_pair[0]
            target_tensor = tensor_pair[1]

            encoder_states, enc_hidden = encoder(input_tensor)
            dec_hidden = enc_hidden
            encoder_states = encoder_states.view(input_tensor.size(0), -1)
            dec_input = torch.tensor([[SOS_index]], dtype=torch.long, device=device)
            loss = 0
            for j in range(target_tensor.size(0)):
                energies = attention(dec_hidden, encoder_states, device)
                context = torch.mm(energies.T, encoder_states).unsqueeze(0)
                dec_output, dec_hidden = decoder(dec_input, context)
                _, idx = dec_output.topk(1, -1)
                if idx.item() == EOS_index:
                    break
                loss += criterion(dec_output[0], target_tensor[j])
                dec_input = idx.squeeze().detach()
            losses.append(loss.item() / target_tensor.size(0))
    return losses


if __name__ == "__main__":
    pickle.dump(input_lang, open('data/input_language.pkl', 'wb'))
    pickle.dump(output_lang, open('data/output_language.pkl', 'wb'))
    pickle.dump(test_pairs, open('data/testing_pairs.pkl', 'wb'))
    encoder.to(device)
    decoder.to(device)
    attn.to(device)
    hidden_state_mapper.to(device)
    losses, valid_losses = train_iters(encoder, decoder, attn)
    plt.figure()
    show_plot(losses)
    show_plot(valid_losses)
    torch.save(encoder.state_dict(), 'model/enc_simple.pt')
    torch.save(decoder.state_dict(), 'model/dec_simple.pt')
    torch.save(attn.state_dict(), 'model/attention.pt')
