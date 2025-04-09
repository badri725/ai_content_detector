# 🛡️ Offensive Comment Detection using Hugging Face Transformers

This project is a Python-based moderation tool that uses a pre-trained transformer model (`unitary/toxic-bert`) from Hugging Face to detect offensive (toxic) comments from a dataset of user inputs.

---

## ⚙️ Setup Instructions

1. Clone this repository or open it in Google Colab.
2. Make sure your environment has Python 3.7+.
3. Install the required packages:

```bash
pip install transformers pandas torch
```

4. Upload your CSV file containing a `comment_text` column to the working directory.

---

## 🚀 How to Use the Script

1. Load the CSV file using `pandas`:

```python
df = pd.read_csv("your_file.csv")
```

2. Run the moderation pipeline:

```python
from transformers import pipeline
moderator = pipeline("text-classification", model="unitary/toxic-bert")

def classify_comment(comment):
    result = moderator(comment[:512])[0]
    return f"{result['label']} ({round(result['score'], 2)})"

df["Moderation_Result"] = df["comment_text"].apply(classify_comment)
```

3. Save the output:

```python
df.to_csv("moderated_comments.csv", index=False)
```

---

## 🧪 Sample Output

| comment_text                | Moderation_Result |
|----------------------------|-------------------|
| You're so dumb and useless | TOXIC (0.98)      |
| Thanks for your help!      | NON_TOXIC (0.97)  |
| Get lost, idiot            | TOXIC (0.95)      |

---

## 📂 Output Files

- `moderated_comments.csv` - Processed file with original comments and moderation result.

---

## ✅ Model Info

- **Model Used**: [unitary/toxic-bert](https://huggingface.co/unitary/toxic-bert)
- **Framework**: Hugging Face `transformers`
- **Runs Offline** in Google Colab

---

Made with ❤️ for the Vyorius AI Moderation Internship Task.