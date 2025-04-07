import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -------------------------------
# Model Definition
# -------------------------------
class SentimentWordVectorModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lambda_theta=0.1, nu=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.lambda_theta = lambda_theta  # regularization weight on theta
        self.nu = nu  # regularization weight on word embeddings (not used in loss below explicitly, but could be added)
        
        # Word embeddings matrix R (each row corresponds to a word's vector)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Per-word bias b (one bias per word)
        self.word_bias = nn.Embedding(vocab_size, 1)
        
        # Sentiment logistic regression parameters: ψ (weight vector) and bias b_c
        self.sentiment_weight = nn.Parameter(torch.randn(embedding_dim))
        self.sentiment_bias = nn.Parameter(torch.zeros(1))
        
    def forward_document(self, doc_word_indices, theta):
        """
        Compute the log probabilities for the words in a document, given a document latent vector theta.
        This implements p(w|θ) = exp(θ^T φ_w + b_w) / sum_{w' in V} exp(θ^T φ_{w'} + b_{w'})
        """
        # Get embeddings and biases for words in the document.
        word_embeds = self.embeddings(doc_word_indices)         # shape: (N, embedding_dim)
        word_biases = self.word_bias(doc_word_indices).squeeze(-1)  # shape: (N,)
        
        # Compute score for each word in the document: θ^T φ_w + b_w
        scores = torch.matmul(word_embeds, theta) + word_biases    # shape: (N,)
        
        # For the softmax denominator, compute logits for every word in the vocabulary.
        all_embeds = self.embeddings.weight                      # shape: (vocab_size, embedding_dim)
        all_biases = self.word_bias.weight.squeeze(-1)           # shape: (vocab_size,)
        logits_all = torch.matmul(all_embeds, theta) + all_biases  # shape: (vocab_size,)
        
        # Use logsumexp for numerical stability.
        log_denominator = torch.logsumexp(logits_all, dim=0)       # scalar
        
        # Log probability for each word in the document.
        log_probs = scores - log_denominator                     # shape: (N,)
        return log_probs
    
    def sentiment_forward(self, word_indices):
        """
        Compute the sentiment prediction for a set of word indices using logistic regression.
        p(s=1|w) = σ(ψ^T φ_w + b_c)
        """
        word_embeds = self.embeddings(word_indices)              # shape: (N, embedding_dim)
        logits = torch.matmul(word_embeds, self.sentiment_weight) + self.sentiment_bias  # shape: (N,)
        probs = torch.sigmoid(logits)                            # shape: (N,)
        return probs

# -------------------------------
# Inference for θ (MAP estimate)
# -------------------------------
def infer_theta(model, doc_word_indices, num_steps=50, lr=0.1):
    """
    For a given document, find the MAP estimate of theta by optimizing
    the unsupervised (semantic) objective.
    """
    # Initialize theta (document latent vector) randomly.
    theta = torch.randn(model.embedding_dim, requires_grad=True)
    optimizer = optim.Adam([theta], lr=lr)
    
    for _ in range(num_steps):
        optimizer.zero_grad()
        # Compute log probabilities for the document words given theta.
        log_probs = model.forward_document(doc_word_indices, theta)
        # Negative log likelihood: we want to maximize log likelihood, so minimize negative.
        nll = -log_probs.sum()
        # Add a quadratic regularization term for theta (i.e., -log p(θ) when p(θ) ~ N(0, I)).
        reg = model.lambda_theta * (theta**2).sum()
        loss = nll + reg
        loss.backward()
        optimizer.step()
    return theta.detach()

# -------------------------------
# Training Loop for the Joint Objective
# -------------------------------
def train_model(model, docs, sentiment_labels, num_epochs=20, theta_infer_steps=50):
    """
    For each document:
      1. Infer theta (MAP estimate) given the document words.
      2. Compute the semantic loss (negative log likelihood) for the document.
      3. Compute the sentiment loss using logistic regression predictions on each word.
      4. Sum losses and update the model parameters.
    """
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for doc, s_label in zip(docs, sentiment_labels):
            # Convert document (list of word indices) to a tensor.
            doc_tensor = torch.tensor(doc, dtype=torch.long)
            # Infer theta for this document.
            theta = infer_theta(model, doc_tensor, num_steps=theta_infer_steps, lr=0.1)
            
            # Semantic loss: negative log likelihood + regularization on theta.
            log_probs = model.forward_document(doc_tensor, theta)
            semantic_loss = -log_probs.sum() + model.lambda_theta * (theta**2).sum()
            
            # Sentiment loss: For each word, predict sentiment and compute BCE loss.
            sentiment_preds = model.sentiment_forward(doc_tensor)
            # Create a target tensor (same sentiment label for each word).
            sentiment_target = torch.full_like(sentiment_preds, float(s_label))
            sentiment_loss = F.binary_cross_entropy(sentiment_preds, sentiment_target, reduction='sum')
            
            # Total loss for this document.
            loss = semantic_loss + sentiment_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} - Total Loss: {total_loss:.4f}")

# -------------------------------
# Example: Dummy Data and Running the Model
# -------------------------------

# Assume a small vocabulary of 10 words.
vocab_size = 10
embedding_dim = 5

# Create our model.
model = SentimentWordVectorModel(vocab_size, embedding_dim, lambda_theta=0.1, nu=0.1)

# Dummy documents (each document is a list of word indices).
# For example, doc1: [1, 3, 5, 2]; doc2: [3, 4, 2, 0]
docs = [
    [1, 3, 5, 2],
    [3, 4, 2, 0],
    [5, 5, 1, 9],
    [0, 2, 8, 3]
]

# Dummy sentiment labels for the documents (1 = positive, 0 = negative)
sentiment_labels = [1, 0, 1, 0]

print("Training the model on dummy data...")
train_model(model, docs, sentiment_labels, num_epochs=20, theta_infer_steps=30)

# -------------------------------
# Testing the Model
# -------------------------------
# Let's pick one document from our dummy set and show the inferred theta and sentiment predictions.
test_doc = docs[0]
doc_tensor = torch.tensor(test_doc, dtype=torch.long)
theta_inferred = infer_theta(model, doc_tensor, num_steps=30, lr=0.1)
log_probs = model.forward_document(doc_tensor, theta_inferred)
sentiment_preds = model.sentiment_forward(doc_tensor)

print("\nTest Document Word Indices:", test_doc)
print("Inferred theta:", theta_inferred)
print("Log probabilities for document words:", log_probs)
print("Sentiment predictions for each word:", sentiment_preds)

