import json
import os
from datetime import datetime


def log_experiment_results(results, log_dir='experiments/results'):
    """Log experiment results to file."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{log_dir}/experiment_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    return filename
