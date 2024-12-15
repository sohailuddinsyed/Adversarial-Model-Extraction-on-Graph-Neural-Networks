"""Different loss function variants for extraction."""
import torch
import torch.nn.functional as F


def cross_entropy_loss(predictions, labels):
    """Standard cross-entropy loss with hard labels."""
    if labels.dim() > 1:
        labels = labels.argmax(dim=1)
    return F.nll_loss(predictions, labels)


def kl_divergence_loss(predictions, soft_labels):
    """KL divergence loss with soft labels (probability distributions)."""
    log_probs = F.log_softmax(predictions, dim=1)
    return F.kl_div(log_probs, soft_labels, reduction='batchmean')


def combined_loss(predictions, soft_labels, true_labels, alpha=0.5):
    """
    Combined loss: balance between fidelity and accuracy.
    L = alpha * fidelity_loss + (1-alpha) * accuracy_loss
    """
    # Fidelity loss (match victim predictions)
    fidelity_loss = kl_divergence_loss(predictions, soft_labels)
    
    # Accuracy loss (match true labels)
    accuracy_loss = cross_entropy_loss(predictions, true_labels)
    
    return alpha * fidelity_loss + (1 - alpha) * accuracy_loss
