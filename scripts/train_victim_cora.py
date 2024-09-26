"""Train victim GCN model on Cora dataset."""
import sys
sys.path.append('.')

from src.models.train_victim import VictimTrainer


def main():
    print("Training victim GCN on Cora dataset...")
    print("=" * 50)
    
    # Initialize trainer
    trainer = VictimTrainer('Cora', hidden_dim=16)
    
    # Train model (140 training samples as per paper)
    history = trainer.train(num_train_per_class=20, epochs=200, verbose=True)
    
    # Save model
    trainer.save_model('models/victim_cora.pth')
    
    # Final evaluation
    train_mask = trainer.loader.get_train_mask(20)
    train_acc = trainer.evaluate(train_mask)
    test_acc = trainer.evaluate(trainer.data.test_mask)
    
    print("\n" + "=" * 50)
    print(f"Final Training Accuracy: {train_acc:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
