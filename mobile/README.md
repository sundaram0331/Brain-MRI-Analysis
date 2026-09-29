# NeonCortrex AI 3.0 — Mobile Android App

Futuristic mobile neuro-radiology diagnostic HUD powered by React Native and Expo.

---

## 🚀 Instant Quickstart (Run on Android in 60 Seconds)

### 1. Install Dependencies
```bash
cd mobile
npm install
```

### 2. Start Expo Development Server
```bash
npx expo start
```

### 3. Open on Your Android Phone
1. Install **Expo Go** from the Google Play Store on your Android phone.
2. Ensure your phone and Mac are connected to the same Wi-Fi network.
3. Open **Expo Go** and scan the QR code printed in your terminal.
4. The cybernetic diagnostic HUD will launch instantly on your device!

---

## ⚡ Connecting to the Diagnostic Backend
1. Start your NeonCortrex FastAPI server:
   ```bash
   cd ../backend
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
2. In the mobile app, tap the **`⚙ API`** button at the top right:
   - For an **Android Emulator**: use `http://10.0.2.2:8000`
   - For a **Physical Android Device**: use your Mac's local Wi-Fi IP (e.g., `http://192.168.1.15:8000`)

---

## 📦 Building a Standalone Android APK (.apk)
To create a standalone installable APK without needing Android Studio:
```bash
# Install EAS CLI
npm install -g eas-cli

# Configure build
npx eas build -p android --profile preview
```
This builds an installable `.apk` file directly in the cloud and gives you a download link to install on any Android device!
