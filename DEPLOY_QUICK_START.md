# 🚀 Quick Start: Deploy on Hugging Face Spaces

## 5-Minute Deployment Guide

### Option 1: One-Click Setup (Easiest)

1. **Create Space on Hugging Face**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Fill in:
     - **Name**: `fatty-liver-classification`
     - **License**: MIT
     - **Space SDK**: Streamlit
   - Click "Create Space"

2. **Copy Files**
   Download from GitHub and upload these files to your Space:
   ```
   app_streamlit.py       (rename to app.py)
   best_model.pth         (trained model file)
   models/
   src/
   requirements.txt
   ```

3. **Push & Deploy**
   ```bash
   git add .
   git commit -m "Deploy fatty liver classification"
   git push
   ```

4. **Access Your App**
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/fatty-liver-classification
   ```

---

### Option 2: Manual Setup

```bash
# Clone your HF Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/fatty-liver-classification
cd fatty-liver-classification

# Copy files from this repo
cp ../fatty-liver/app_streamlit.py app.py
cp ../fatty-liver/best_model.pth .
cp -r ../fatty-liver/models .
cp -r ../fatty-liver/src .
cp ../fatty-liver/requirements_streamlit.txt requirements.txt

# Push to Hugging Face
git add .
git commit -m "Add Fatty Liver Classification"
git push origin main
```

---

## 📁 Required Files

```
your-space/
├── app.py                    # Streamlit app (rename from app_streamlit.py)
├── best_model.pth           # Trained model (~95 MB)
├── requirements.txt         # Dependencies
├── models/
│   ├── __init__.py
│   └── siamese_net.py       # Model architecture
└── src/
    ├── __init__.py
    └── data_loader.py       # Data preprocessing
```

---

## ✅ What to Expect

✨ **Deployment Time**: 3-5 minutes
✨ **First Load**: 1-2 minutes (model loading)
✨ **File Size**: ~95 MB (model is large)
✨ **Uptime**: Always available (free tier)

---

## 🔑 Important Notes

1. **Model File Size**: 95 MB is acceptable on HF Spaces
2. **First Load**: App may take 1-2 min to load model
3. **Storage**: HF Spaces provides 50 GB free space
4. **Compute**: Free tier is sufficient for inference

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Module not found" | Ensure `models/` and `src/` folders are copied |
| "Model not found" | Check `best_model.pth` is in root directory |
| App won't load | Wait 2 minutes, then refresh page |
| File not found | Verify file names match exactly |

---

## 📊 Features Available

- ✅ Multi-class (5-grade) classification
- ✅ Binary (Normal/Abnormal) classification
- ✅ Real-time predictions
- ✅ Confidence scores
- ✅ Probability distribution
- ✅ Image validation
- ✅ Clinical recommendations

---

## 🔐 Important: Privacy & Security

- Models run **locally** (no data sent to servers)
- Images are **not stored** after processing
- **Private** by default on HF Spaces
- Can make **public** for sharing

---

## 📞 Need Help?

1. Check HF Spaces docs: https://huggingface.co/docs/hub/spaces
2. Visit GitHub repo for updates
3. Review `HF_DEPLOYMENT.md` for detailed guide

---

**Happy Deploying! 🚀**
