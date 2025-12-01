import matplotlib.pyplot as plt
import pickle
import os

# If you have a saved training history file
try:
    with open('models/training_history.pkl', 'rb') as f:
        history = pickle.load(f)
    
    # Extract metrics
    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    
    epochs = range(1, len(accuracy) + 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(epochs, accuracy, 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, val_accuracy, 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('results/training_history_graphs.png', dpi=300, bbox_inches='tight')
    print("✅ Graphs saved to: results/training_history_graphs.png")
    
    # Show the plots
    plt.show()
    
    # Print final metrics
    print(f"\n📊 Final Training Accuracy: {accuracy[-1]:.4f} ({accuracy[-1]*100:.2f}%)")
    print(f"📊 Final Validation Accuracy: {val_accuracy[-1]:.4f} ({val_accuracy[-1]*100:.2f}%)")
    print(f"📊 Final Training Loss: {loss[-1]:.4f}")
    print(f"📊 Final Validation Loss: {val_loss[-1]:.4f}")

except FileNotFoundError:
    print("❌ Training history file not found!")
    print("You need to save training history during model training.")
    print("\nAdd this to your training script:")
    print("""
    # After model.fit()
    import pickle
    with open('models/training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    """)

