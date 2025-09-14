import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch import softmax
from torch.utils.data import DataLoader, Dataset



class GloVeModel(nn.Module):
    """Implement GloVe model with Pytorch
    """

    def __init__(self, embedding_size, context_size, vocab_size): #, min_occurrance=1, x_max=100, alpha=3 / 4):
        super(GloVeModel, self).__init__()

        self.embedding_size = embedding_size
        self.context_size = context_size
        self.vocab_size = vocab_size
        # self.min_occurrance = min_occurrance
        # self.x_max = x_max
        # self.alpha = alpha
        dtype = torch.float32
        self._focal_embeddings = nn.Embedding(vocab_size, embedding_size).type(dtype)
        self._context_embeddings = nn.Embedding(vocab_size, embedding_size).type(dtype)
        self._focal_biases = nn.Embedding(vocab_size, 1).type(dtype)
        self._context_biases = nn.Embedding(vocab_size, 1).type(dtype)
        self._glove_dataset = None
        self.VOCAB_TENSOR = torch.arange(self.vocab_size).long()
        for params in self.parameters():
            init.uniform_(params, a=-1, b=1)

    def train(self, num_epoch, batch_size=512, learning_rate=0.05, loop_interval=10):
        """Training GloVe model

        Args:
            num_epoch (int): number of epoch
            device (str): cpu or gpu
            batch_size (int, optional): Defaults to 512.
            learning_rate (float, optional): Defaults to 0.05. learning rate for Adam optimizer
            batch_interval (int, optional): Defaults to 100. interval time to show average loss

        Raises:
            NotFitToCorpusError: if the model is not fit by corpus, the error will be raise
        """
        # basic training setting
        device = next(self.parameters()).device
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        glove_dataloader = DataLoader(self._glove_dataset, batch_size)
        total_loss = 0
        context_input = self.VOCAB_TENSOR.to(device)
        for epoch in range(num_epoch):
            for idx, batch in enumerate(glove_dataloader):
                optimizer.zero_grad()
                counts = batch.to(device)
                bsize = batch.shape[0]
                focal_input = torch.arange(idx * batch_size, idx * batch_size + bsize).long().to(device)
                loss, orto_loss = self._loss(focal_input, context_input, counts)
                total_loss += loss.item()
                if idx % loop_interval == 0:
                    avg_loss = total_loss / loop_interval
                    print("epoch: {}, current step: {}, average loss: {}    {}".format(
                        epoch, idx, avg_loss, orto_loss))
                    total_loss = 0

                loss.backward()
                optimizer.step()

        print("finish glove vector training")

    def get_coocurrance_matrix(self):
        """ Return co-occurance matrix for saving

        Returns:
            list: list itam (word_idx1, word_idx2, cooccurances)
        """

        return self._glove_dataset._coocurrence_matrix

    def embedding_for_tensor(self, tokens):
        if not torch.is_tensor(tokens):
            raise ValueError("the tokens must be pytorch tensor object")

        return self._focal_embeddings(tokens) + self._focal_biases(tokens) + self._context_embeddings(tokens) + self._context_biases(tokens)

    @torch.no_grad()
    def predict(self, focal_input, temperature=2):
        device = focal_input.device
        context_input = self.VOCAB_TENSOR.to(device)
        focal_embed = torch.mean(self._focal_embeddings(focal_input), axis=1)
        context_embed = self._context_embeddings(context_input)
        # count weight factor
        embedding_products = focal_embed[:, None, :] * context_embed[None, :, :]
        scores = torch.sum((context_embed - embedding_products) ** 2, axis=2)
        # 1) get top-k values and their global indices
        values, indices = torch.topk(scores, k=10, largest=False, dim=1)  # values: (k,), indices: (k,)

        # 2) convert to probabilities using softmax with temperature
        probs = softmax(values / temperature, dim=1)  # (k,)

        # 3) sample one index from top-k according to probs
        # torch.multinomial expects probs to sum to 1
        chosen_pos = torch.multinomial(probs[:], num_samples=1)  # 0..k-1
        chosen_global_index = indices[:, chosen_pos][:]

        return chosen_global_index

    def _loss(self, focal_input, context_input, coocurrence_count):
        # x_max, alpha = self.x_max, self.alpha

        focal_embed = self._focal_embeddings(focal_input)
        context_embed = self._context_embeddings(context_input)
        focal_bias = self._focal_biases(focal_input)
        context_bias = self._context_biases(context_input)

        # count weight factor
        embedding_products = torch.sum(focal_embed[:, :, None] * context_embed.T[None, :, :], dim=1)
        mask = torch.zeros_like(coocurrence_count>0)
        log_cooccurrences = torch.zeros(coocurrence_count.shape, device=coocurrence_count.device)
        log_cooccurrences[mask] = torch.log(coocurrence_count[mask])

        distance_expr = (embedding_products + focal_bias + context_bias.T + log_cooccurrences) ** 2

        single_losses = coocurrence_count * distance_expr
        mean_loss = torch.mean(single_losses)

        x = (self._context_embeddings.weight + self._focal_biases.weight)/2
        loss = torch.norm(torch.eye(x.shape[1], device=x.device) - x.T @ x)

        return mean_loss + loss * 1e-5, loss


class GloVeDataSet(Dataset):

    def __init__(self, coocurrence_matrix):
        self._coocurrence_matrix = coocurrence_matrix

    def __getitem__(self, index):
        return self._coocurrence_matrix[index]

    def __len__(self):
        return len(self._coocurrence_matrix)

