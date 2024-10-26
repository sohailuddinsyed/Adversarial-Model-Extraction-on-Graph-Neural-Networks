import torch


class FidelityMeasure:
    """
    Measure fidelity between victim and extracted models.
    Fidelity = percentage of nodes where both models agree.
    """
    
    @staticmethod
    def compute_fidelity(victim_predictions, extracted_predictions):
        """
        Compute fidelity between victim and extracted model predictions.
        
        Args:
            victim_predictions: Victim model predictions [num_nodes]
            extracted_predictions: Extracted model predictions [num_nodes]
            
        Returns:
            Fidelity score (0 to 1)
        """
        if victim_predictions.dim() > 1:
            victim_predictions = victim_predictions.argmax(dim=1)
        if extracted_predictions.dim() > 1:
            extracted_predictions = extracted_predictions.argmax(dim=1)
        
        agreement = (victim_predictions == extracted_predictions).float()
        fidelity = agreement.mean().item()
        
        return fidelity
    
    @staticmethod
    def compute_fidelity_per_class(victim_predictions, extracted_predictions, labels, num_classes):
        """
        Compute fidelity per class.
        
        Args:
            victim_predictions: Victim predictions
            extracted_predictions: Extracted predictions
            labels: True labels
            num_classes: Number of classes
            
        Returns:
            Dictionary with per-class fidelity
        """
        if victim_predictions.dim() > 1:
            victim_predictions = victim_predictions.argmax(dim=1)
        if extracted_predictions.dim() > 1:
            extracted_predictions = extracted_predictions.argmax(dim=1)
        
        fidelity_per_class = {}
        
        for c in range(num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                agreement = (victim_predictions[mask] == extracted_predictions[mask]).float()
                fidelity_per_class[c] = agreement.mean().item()
            else:
                fidelity_per_class[c] = 0.0
        
        return fidelity_per_class
    
    @staticmethod
    def compute_accuracy(predictions, labels):
        """
        Compute accuracy against true labels.
        
        Args:
            predictions: Model predictions
            labels: True labels
            
        Returns:
            Accuracy score
        """
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=1)
        
        correct = (predictions == labels).float()
        accuracy = correct.mean().item()
        
        return accuracy
