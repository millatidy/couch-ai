import numpy as np
import pickle

from keras.models import Model
from keras.layers import Input, LSTM, Dense

class Chatbot:

    def __init__(self):
        self.input_texts = 35
        self.latent_dim = 256
        self.max_encoder_seq_length = 80
        self.max_decoder_seq_length = 254
        self.charecter_index = None
        self.uniqie_tokens = None
        self.reverse_target_char_index = None
        self.encoder_inputs = None
        self.encoder = None
        self.encoder_outputs = None
        self.state_h = None
        self.state_c = None
        self.encoder_states = None
        self.decoder_inputs = None
        self.decoder_lstm = None
        self.decoder_outputs = None
        self.decoder_dense = None
        self.decoder_outputs = None
        self.model = None
        self.encoder_model = None
        self.decoder_state_input_h = None
        self.decoder_state_input_c = None
        self.decoder_states_inputs = None
        self.decoder_model = None


    def load_obj(self, name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)


    def set_char_indexs(self):
        self.charecter_index = self.load_obj('charecter_index')
        self.uniqie_tokens = self.load_obj('uniqie_tokens')
        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.charecter_index.items())


    def create_base_model(self):
        # Define an input sequence and process it.
        self.encoder_inputs = Input(shape=(None, self.uniqie_tokens))
        self.encoder = LSTM(self.latent_dim, return_state=True)
        self.encoder_outputs, self.state_h, self.state_c = self.encoder(self.encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        self.encoder_states = [self.state_h, self.state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        self.decoder_inputs = Input(shape=(None, self.uniqie_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        self.decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        self.decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs,
                                             initial_state=self.encoder_states)
        self.decoder_dense = Dense(self.uniqie_tokens, activation='softmax')
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)


    def load_weights(self):
        print('Loading model...')
        self.model.load_weights('new_data_proc_weights1.h5')
        print ('Model loaded!')


    def create_inference_model(self):
        # Define sampling models
        self.encoder_model = Model(self.encoder_inputs, self.encoder_states)

        self.decoder_state_input_h = Input(shape=(self.latent_dim,))
        self.decoder_state_input_c = Input(shape=(self.latent_dim,))
        self.decoder_states_inputs = [self.decoder_state_input_h, self.decoder_state_input_c]
        self.decoder_outputs, self.state_h, self.state_c = self.decoder_lstm(
            self.decoder_inputs, initial_state=self.decoder_states_inputs)
        self.decoder_states = [self.state_h, self.state_c]
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)
        self.decoder_model = Model(
            [self.decoder_inputs] + self.decoder_states_inputs,
            [self.decoder_outputs] + self.decoder_states)


    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.uniqie_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.charecter_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
               len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.uniqie_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence


    def chat(self, input_text):
        encoder_input_data = np.zeros(
            (1, self.max_encoder_seq_length, self.uniqie_tokens), dtype='float32')
        for t, char in enumerate(input_text.lower()):
            encoder_input_data[0,t,self.charecter_index[char]] = 1.
        input_seq = encoder_input_data
        decoded_sentence = self.decode_sequence(input_seq)
        return decoded_sentence

    def initialize(self):
        self.set_char_indexs()
        self.create_base_model()
        self.load_weights()
        self.create_inference_model()


    def main(self):
        self.initialize()
        while(True):
            input_text = input("Enter message: ")
            decoded_sentence = self.chat(input_text)
            print('-')
            print('Decoded sentence:', decoded_sentence)


if __name__ == '__main__':
    bot = Chatbot()
    bot.main()
