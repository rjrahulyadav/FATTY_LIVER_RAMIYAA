# 🏥 Fatty Liver Classification - Hugging Face Deployment

## 📋 Deployment Instructions

### Step 1: Create a New Space on Hugging Face
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. **Name**: `fatty-liver-classification` (or your choice)
4. **License**: Choose appropriate license
5. **Space SDK**: Select **Streamlit**
6. Click "Create Space"

### Step 2: Clone the Space Repository
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/fatty-liver-classification
cd fatty-liver-classification
```

### Step 3: Copy Files to Space
Copy these files from this repository:
- `app_streamlit.py` → rename to `app.py`
- `best_model.pth` (trained model)
- `models/` (folder)
- `src/` (folder)
- `requirements_streamlit.txt` → rename to `requirements.txt`

### Step 4: Push to Hugging Face
```bash
git add .
git commit -m "Add Fatty Liver Classification app"
git push
```

### ✅ Done!
Your app will automatically deploy. Access it at:
`https://huggingface.co/spaces/YOUR_USERNAME/fatty-liver-classification`

---

## 📊 Features

✨ **Multi-class Classification**: 5-grade fatty liver assessment
- Normal
- Grade-I (5-35% fat)
- Grade-II (35-65% fat)
- Grade-III (>65% fat)
- CLD (Chronic Liver Disease)

✨ **Binary Classification**: Simple Normal/Abnormal
- Normal
- Abnormal (any pathology)

✨ **High Accuracy**: 99%+ validation accuracy

✨ **User-Friendly Interface**: 
- Easy image upload
- Real-time predictions
- Probability distribution
- Clinical recommendations

---

## 🛠️ Technical Details

- **Framework**: Streamlit
- **Model**: ResNet-50 Siamese Network
- **Backend**: PyTorch
- **Deployment**: Hugging Face Spaces (Free)

---

## ⚠️ Important Disclaimer

**This tool is for research and educational purposes ONLY.**
- Not for clinical diagnosis
- Always consult qualified medical professionals
- AI results must be verified by radiologists
- Use at your own responsibility

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section below
2. Consult medical imaging experts
3. Open an issue on the repository

---

## 🔧 Troubleshooting

### Model not found error
- Ensure `best_model.pth` is in the Space root directory
- Check file size (~95MB)

### Import errors
- Verify all folders are copied: `models/`, `src/`
- Check `requirements.txt` is properly formatted

### Slow loading
- Streamlit may take 1-2 min on first load
- Space may auto-sleep after 30 min inactivity

---

## 📈 Future Improvements

- [ ] Real-time model retraining
- [ ] Multiple model options
- [ ] Batch processing
- [ ] Export detailed reports
- [ ] Integration with medical databases

---

Made with ❤️ for Medical Imaging Analysis
