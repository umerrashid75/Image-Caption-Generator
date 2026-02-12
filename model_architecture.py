
import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, input_size=2048, embed_size=256):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(input_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, features):
        # Features: (batch, 2048)
        # Output: (batch, embed_size)
        embed = self.linear(features)
        embed = self.bn(embed)
        embed = self.relu(embed)
        return self.dropout(embed)

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # features: (batch, embed_size)
        # captions: (batch, seq_len)
        
        # Embed the captions
        embeddings = self.dropout(self.embed(captions))
        # (batch, seq_len, embed_size)
        
        # Concatenate features as the first step of the sequence
        # We unsqueeze features to be (batch, 1, embed_size)
        features = features.unsqueeze(1)
        
        # Input to LSTM: [features, caption_word_1, caption_word_2, ...]
        # We usually discard the last <END> token for training target alignment
        inputs = torch.cat((features, embeddings[:, :-1, :]), dim=1)
        
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs

class Seq2Seq(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(2048, embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, features, captions):
        encoded_features = self.encoder(features)
        outputs = self.decoder(encoded_features, captions)
        return outputs
    
    def caption_image(self, features, vocabulary, max_length=20):
        # Greedy Search
        result_caption = []
        
        with torch.no_grad():
            x = self.encoder(features).unsqueeze(1) # (1, 1, embed)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(1))
                predicted = output.argmax(1)
                
                result_caption.append(predicted.item())
                
                x = self.decoder.embed(predicted).unsqueeze(1)

                if vocabulary.itos[predicted.item()] == "<END>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]

    def caption_image_beam_search(self, features, vocabulary, beam_size=3, max_length=20):
        # Beam Search
        k = beam_size
        start_token = vocabulary.stoi["<START>"]
        end_token = vocabulary.stoi["<END>"]
        device = features.device
        
        with torch.no_grad():
            # Initial encode: Ensure simple (1, 2048) batch dim
            if features.dim() == 1:
                features = features.unsqueeze(0)
                
            # Encoder output -> (1, embed) -> (1, 1, embed)
            encoded_feat = self.encoder(features).unsqueeze(1)
            
            # Initial step to get first hidden state
            # h, c shape: (num_layers, batch=1, hidden_size)
            h, c = self.decoder.lstm(encoded_feat, None)[1]
            
            # Beams store: (cumulative_score, list_of_token_indices, (h, c))
            beams = [(0, [start_token], (h, c))]
            
            for _ in range(max_length):
                candidates = []
                
                # Expand each beam
                for score, seq, (h, c) in beams:
                    if seq[-1] == end_token:
                        candidates.append((score, seq, (h, c)))
                        continue
                    
                    # Prepare input for next step
                    last_token_idx = seq[-1]
                    # embed input shape: (1) -> output (1, embed_size)
                    # We want (batch=1, seq=1, embed_size) -> (1, 1, embed_size)
                    x = self.decoder.embed(torch.tensor([last_token_idx]).to(device)).unsqueeze(1) 
                    
                    # Ensure hidden states are contiguous tensors for LSTM
                    h_in = h.contiguous()
                    c_in = c.contiguous()
                    
                    # LSTM Step
                    # output: (batch, seq_len, hidden)
                    # new_states: ((num_layers, batch, hidden), (num_layers, batch, hidden))
                    out, (h_next, c_next) = self.decoder.lstm(x, (h_in, c_in))
                    
                    # Predict next word
                    output = self.decoder.linear(out.squeeze(1)) # (1, vocab)
                    probs = torch.nn.functional.log_softmax(output, dim=1)
                    topk_probs, topk_ids = probs.topk(k)
                    
                    for i in range(k):
                        word_idx = topk_ids[0][i].item()
                        word_prob = topk_probs[0][i].item()
                        new_score = score + word_prob
                        new_seq = seq + [word_idx]
                        candidates.append((new_score, new_seq, (h_next, c_next)))
                
                # Order all candidates by score
                ordered = sorted(candidates, key=lambda x: x[0], reverse=True)
                beams = ordered[:k]
                
                # If all top k beams ended, stop early
                if all(b[1][-1] == end_token for b in beams):
                    break
            
            best_seq = beams[0][1]
            return [vocabulary.itos[idx] for idx in best_seq if idx not in [start_token, end_token]]
