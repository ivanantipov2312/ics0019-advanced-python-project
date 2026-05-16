import matplotlib.pyplot as plt  # installed automatically with seaborn
import seaborn as sns
from sklearn.metrics import confusion_matrix

labels = ["DoS", "Normal", "Probe", "R2L", "U2R"]
cm = confusion_matrix(["DoS", "Normal", "Probe", "R2L", "U2R"], ["DoS", "Normal", "Probe", "R2L", "U2R"], labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
