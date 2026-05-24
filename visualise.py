import joblib
import matplotlib.pyplot as plt  # installed automatically with seaborn
import seaborn as sns
from sklearn.metrics import confusion_matrix

y_test = joblib.load("y_test.pkl")
y_pred = joblib.load("y_pred.pkl")
label_encoder_y = joblib.load("label_encoder.pkl")

y_test_labels = label_encoder_y.inverse_transform(y_test)
y_pred_labels = label_encoder_y.inverse_transform(y_pred)

labels = ["DoS", "Normal", "Probe", "R2L", "U2R"]

cm = confusion_matrix(y_test_labels, y_pred_labels, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
