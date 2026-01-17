# 🚀 Complete Deployment Guide: Hugging Face Spaces

## Overview

This guide walks you through deploying your Fatty Liver Classification app on **Hugging Face Spaces** - completely FREE with no credit card needed.

**Why HF Spaces?**
- ✅ 100% Free (no paid tier needed)
- ✅ Unlimited storage (50GB per Space)
- ✅ Always available (no spin-down)
- ✅ Auto-deploys from git push
- ✅ Great for ML/AI projects

---

## 📋 Prerequisites

Before starting, you need:

1. **Hugging Face Account**
   - Sign up: https://huggingface.co/join
   - Free account

2. **Git Installed**
   - Download: https://git-scm.com/download/win
   - Verify: `git --version` in terminal

3. **HF Access Token**
   - Go to: https://huggingface.co/settings/tokens
   - Click "New token"
   - Name: "HF Spaces Deployment"
   - Role: "write"
   - Copy the token (you'll need it)

---

## 🎯 Method 1: Automated Deployment (Easiest)

### Run the Deployment Script

```bash
cd "C:\Users\zayac\OneDrive\Desktop\fatty liver"
python deploy_hf_spaces.py
```

The script will:
1. ✅ Check prerequisites
2. ✅ Ask for your HF credentials
3. ✅ Clone your Space repository
4. ✅ Copy all required files
5. ✅ Push to Hugging Face
6. ✅ Show your deployed app URL

**That's it!** Your app will be live in 2-3 minutes.

---

## 🎯 Method 2: Manual Deployment

### Step 1: Create a Space on Hugging Face

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"** button
3. Fill in:
   - **Owner**: Select your account
   - **Space name**: `fatty-liver-classification`
   - **License**: MIT (or your choice)
   - **Space SDK**: Select **Streamlit**
   - Click **"Create Space"**

### Step 2: Clone the Space Repository

Open terminal and run:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/fatty-liver-classification
cd fatty-liver-classification
```

Replace `YOUR_USERNAME` with your actual HF username.

### Step 3: Copy Required Files

Copy these files from your GitHub repo to the Space folder:

```bash
# From the GitHub repo folder, copy:
cp app_streamlit.py path\to\space\app.py
cp best_model.pth path\to\space\
cp -r models path\to\space\
cp -r src path\to\space\
cp requirements_streamlit.txt path\to\space\requirements.txt
```

### Step 4: Create .gitignore (Optional)

```bash
cd path\to\space
```

Create a file named `.gitignore` with:
```
__pycache__/
*.pyc
.env
.streamlit/
.venv/
```

### Step 5: Commit and Push

```bash
git add .
git commit -m "Deploy Fatty Liver Classification app"
git push
```

Enter your HF token when prompted (copy from settings/tokens).

### Step 6: Wait for Deployment

- Go to: https://huggingface.co/spaces/YOUR_USERNAME/fatty-liver-classification
- Wait 2-3 minutes for build
- First load may take 1-2 minutes (model loading)

✅ **Done!** Your app is live!

---

## 📁 File Structure on Space

Your Space should look like:

```
fatty-liver-classification/
├── app.py                    (renamed from app_streamlit.py)
├── best_model.pth           (model file, ~95 MB)
├── requirements.txt         (dependencies)
├── models/
│   ├── __init__.py
│   └── siamese_net.py
├── src/
│   ├── __init__.py
│   └── data_loader.py
└── .gitignore
```

---

## 🔑 Configuration Files

### requirements.txt
Should contain:
```
streamlit==1.28.1
torch==2.0.1
torchvision==0.15.2
Pillow==10.0.0
numpy==1.24.3
scipy==1.11.2
scikit-image==0.21.0
tqdm==4.66.1
```

### app.py
Should be the renamed `app_streamlit.py` file

---

## 🧪 Testing Your Deployment

1. **Access your Space**
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/fatty-liver-classification
   ```

2. **Wait for first load** (1-2 minutes)

3. **Upload a test image**
   - Use any liver ultrasound image
   - App will show prediction

4. **Check results**
   - Confidence should be 70%+
   - Probability distribution shows breakdown

---

## 🛠️ Troubleshooting

### Problem: "Module not found" Error

**Solution:**
- Verify `models/` and `src/` folders exist
- Check file names match exactly (case-sensitive)
- Push changes again: `git add . && git commit -m "Fix imports" && git push`

### Problem: "Model not found" Error

**Solution:**
- Ensure `best_model.pth` is in root directory
- Check file size (~95 MB)
- Push again if missing

### Problem: App takes too long to load

**Solution:**
- First load always takes 1-2 min (model initialization)
- Subsequent loads are faster (~10-30 seconds)
- File size is 95 MB, which is acceptable

### Problem: "Permission denied" when pushing

**Solution:**
1. Go to https://huggingface.co/settings/tokens
2. Create new token with "write" access
3. Use this new token

### Problem: Cannot clone repository

**Solution:**
- Ensure Space exists on HF
- Check username is correct
- Verify token has proper permissions
- Try: `git clone https://huggingface.co/spaces/USERNAME/SPACENAME`

---

## 🔐 Making Your Space Private/Public

1. Go to your Space settings
2. Under "Visibility"
3. Choose "Private" or "Public"
4. Save

**Recommendation**: Keep "Private" for production apps, or "Public" for sharing.

---

## 📈 Monitoring & Updates

### View Space Logs
- Click the "Logs" tab in your Space
- Useful for debugging

### Push Updates
Any changes you push will auto-deploy:
```bash
# Make changes locally
git add .
git commit -m "Update description or fix"
git push
# Space rebuilds automatically
```

### Monitor Usage
- Go to Space settings
- View access statistics

---

## ⚡ Performance Tips

1. **Model Caching**
   - Streamlit automatically caches model
   - First load: 1-2 min
   - Subsequent: 30 sec

2. **Image Optimization**
   - App handles image resizing
   - Supports JPG, PNG, BMP

3. **Inference Speed**
   - CPU inference: ~2-5 seconds per image
   - GPU (if available): <1 second

---

## 🔗 Useful Links

- **Hugging Face Spaces**: https://huggingface.co/spaces
- **HF Documentation**: https://huggingface.co/docs/hub/spaces
- **Streamlit Docs**: https://docs.streamlit.io/
- **Your Repository**: https://github.com/rjrahulyadav/FATTY_LIVER_RAMIYAA

---

## 📞 Support

If you encounter issues:

1. **Check the logs**
   - Click "Logs" tab in Space

2. **Verify file structure**
   - All required files present?
   - Correct names?

3. **Review this guide**
   - Common issues section above

4. **Consult HF docs**
   - https://huggingface.co/docs/hub/spaces

---

## ✅ Deployment Checklist

- [ ] HF account created
- [ ] Access token generated and saved
- [ ] Git installed and verified
- [ ] Space created on HF
- [ ] Repository cloned locally
- [ ] Files copied (app.py, model, models/, src/, requirements.txt)
- [ ] .gitignore created (optional)
- [ ] Changes committed
- [ ] Pushed to HF
- [ ] Space URL accessible
- [ ] First load completed
- [ ] Test image uploaded and predicted
- [ ] Results look correct

---

## 🎉 You're Done!

Your Fatty Liver Classification app is now live on the internet, completely FREE, and accessible to anyone with your Space URL.

**Share your Space URL to:**
- Collaborate with colleagues
- Demo to stakeholders
- Get feedback from users
- Showcase your work

---

**Happy Deploying! 🚀**

*For issues or questions, feel free to open an issue on GitHub or check HF documentation.*
