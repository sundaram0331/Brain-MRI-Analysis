import React, { useState } from 'react';
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Alert,
  SafeAreaView,
  StatusBar,
  Dimensions,
  TextInput
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as Haptics from 'expo-haptics';

const { width } = Dimensions.get('window');

const TUMOR_THEMES = {
  Glioma: { color: '#FF4D4D', bg: '#2A1010', border: '#FF4D4D' },
  Meningioma: { color: '#FF8C00', bg: '#2A1C10', border: '#FF8C00' },
  Pituitary: { color: '#A855F7', bg: '#23122E', border: '#A855F7' },
  'No Tumor': { color: '#22C55E', bg: '#0D2618', border: '#22C55E' },
};

export default function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [activeLayer, setActiveLayer] = useState('original');
  const [backendUrl, setBackendUrl] = useState('http://10.176.98.158:8000'); // Auto-detected Wi-Fi IP
  const [showConfig, setShowConfig] = useState(false);

  // Request camera and gallery permissions
  const pickFromGallery = async () => {
    try {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Camera roll access is needed to select MRI scans.');
        return;
      }
      const res = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.9,
        base64: true,
      });

      if (!res.canceled && res.assets && res.assets[0]) {
        setSelectedImage(res.assets[0]);
        setResult(null);
      }
    } catch (e) {
      Alert.alert('Error', e.message);
    }
  };

  const takePhoto = async () => {
    try {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Camera access is required to capture MRI scans.');
        return;
      }
      const res = await ImagePicker.launchCameraAsync({
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.9,
        base64: true,
      });

      if (!res.canceled && res.assets && res.assets[0]) {
        setSelectedImage(res.assets[0]);
        setResult(null);
      }
    } catch (e) {
      Alert.alert('Error', e.message);
    }
  };

  const runAnalysis = async () => {
    if (!selectedImage) {
      Alert.alert('No Image', 'Please capture or select an MRI scan first.');
      return;
    }

    setLoading(true);
    try {
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
      
      const payload = {
        image: selectedImage.base64 ? `data:image/jpeg;base64,${selectedImage.base64}` : selectedImage.uri,
        filename: 'mobile_mri.jpg',
      };

      const res = await fetch(`${backendUrl}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
      setActiveLayer('original');

      // Heavy celebratory or warning haptic feedback on Android
      await Haptics.notificationAsync(
        data.predicted_class === 'No Tumor' 
          ? Haptics.NotificationFeedbackType.Success 
          : Haptics.NotificationFeedbackType.Warning
      );
    } catch (err) {
      Alert.alert(
        'Backend Connection',
        `Could not reach API at ${backendUrl}.\n\nEnsure backend is running: "uvicorn main:app --host 0.0.0.0 --port 8000"\n\nDetails: ${err.message}`
      );
    } finally {
      setLoading(false);
    }
  };

  const currentTheme = result ? (TUMOR_THEMES[result.predicted_class] || { color: '#00F0FF', bg: '#081726', border: '#00F0FF' }) : null;

  const getImageForLayer = () => {
    if (!result) return selectedImage?.uri;
    const imgs = result.images || {};
    switch (activeLayer) {
      case 'heatmap':
        return imgs.gradcam ? `data:image/jpeg;base64,${imgs.gradcam}` : result.heatmap_image ? `data:image/jpeg;base64,${result.heatmap_image}` : selectedImage?.uri;
      case 'segmentation':
        return imgs.segmentation ? `data:image/jpeg;base64,${imgs.segmentation}` : result.segmentation_image ? `data:image/jpeg;base64,${result.segmentation_image}` : selectedImage?.uri;
      case 'contour':
        return imgs.contour ? `data:image/jpeg;base64,${imgs.contour}` : result.contour_image ? `data:image/jpeg;base64,${result.contour_image}` : selectedImage?.uri;
      case 'bbox':
        return imgs.bbox ? `data:image/jpeg;base64,${imgs.bbox}` : result.bbox_image ? `data:image/jpeg;base64,${result.bbox_image}` : selectedImage?.uri;
      default:
        return imgs.original ? `data:image/jpeg;base64,${imgs.original}` : selectedImage?.uri;
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#080C14" />
      
      {/* ── TOP SCI-FI APP HEADER ── */}
      <View style={styles.header}>
        <View>
          <Text style={styles.brandTitle}>NEONCORTREX <Text style={styles.brandAccent}>3.0</Text></Text>
          <Text style={styles.brandSub}>NEURO-RADIOLOGY MOBILE HUD</Text>
        </View>
        <TouchableOpacity 
          style={styles.configBtn} 
          onPress={() => setShowConfig(!showConfig)}
        >
          <Text style={styles.configBtnText}>⚙ API</Text>
        </TouchableOpacity>
      </View>

      {/* ── API CONFIGURATION BAR (IF TOGGLED) ── */}
      {showConfig && (
        <View style={styles.configBox}>
          <Text style={styles.configLabel}>BACKEND HOST URL:</Text>
          <TextInput
            style={styles.configInput}
            value={backendUrl}
            onChangeText={setBackendUrl}
            autoCapitalize="none"
            placeholder="http://192.168.1.X:8000"
            placeholderTextColor="#556987"
          />
          <Text style={styles.configHint}>Tip: For physical Android device, use your Mac's LAN Wi-Fi IP address.</Text>
        </View>
      )}

      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* ── MRI VIEWFINDER / SCAN DISPLAY ── */}
        <View style={[styles.scanViewport, currentTheme && { borderColor: currentTheme.border }]}>
          {selectedImage ? (
            <Image 
              source={{ uri: getImageForLayer() }} 
              style={styles.mriImage} 
              resizeMode="contain" 
            />
          ) : (
            <View style={styles.emptyPrompt}>
              <Text style={styles.emptyReticle}>[ + ]</Text>
              <Text style={styles.emptyTitle}>NO CRANIAL SCAN LOADED</Text>
              <Text style={styles.emptySubtitle}>Snap lightbox film or choose from phone gallery</Text>
            </View>
          )}

          {/* Hologram Reticle Corner Brackets */}
          <View style={[styles.cornerBracket, styles.topLeft, currentTheme && { borderColor: currentTheme.color }]} />
          <View style={[styles.cornerBracket, styles.topRight, currentTheme && { borderColor: currentTheme.color }]} />
          <View style={[styles.cornerBracket, styles.bottomLeft, currentTheme && { borderColor: currentTheme.color }]} />
          <View style={[styles.cornerBracket, styles.bottomRight, currentTheme && { borderColor: currentTheme.color }]} />
        </View>

        {/* ── CAMERA & GALLERY ACTION BUTTONS ── */}
        <View style={styles.actionRow}>
          <TouchableOpacity style={styles.toolBtn} onPress={takePhoto}>
            <Text style={styles.toolBtnIcon}>📷</Text>
            <Text style={styles.toolBtnText}>CAMERA SCAN</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.toolBtn} onPress={pickFromGallery}>
            <Text style={styles.toolBtnIcon}>🖼️</Text>
            <Text style={styles.toolBtnText}>GALLERY</Text>
          </TouchableOpacity>
        </View>

        {/* ── INFERENCE EXECUTION BUTTON ── */}
        {selectedImage && !result && (
          <TouchableOpacity 
            style={[styles.analyzeBtn, loading && styles.analyzeBtnDisabled]} 
            onPress={runAnalysis}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="#080C14" size="small" />
            ) : (
              <Text style={styles.analyzeBtnText}>⚡ RUN NEURAL DIAGNOSIS</Text>
            )}
          </TouchableOpacity>
        )}

        {/* ── DIAGNOSTIC RESULTS HUD ── */}
        {result && (
          <View style={styles.resultsContainer}>
            {/* Primary Classification Banner */}
            <View style={[styles.diagCard, { borderColor: currentTheme.border, backgroundColor: currentTheme.bg }]}>
              <View style={styles.cardHeaderRow}>
                <Text style={[styles.classBadge, { color: currentTheme.color }]}>
                  {result.predicted_class?.toUpperCase()}
                </Text>
                <Text style={styles.confBadge}>
                  {result.confidence}% CONFIDENCE
                </Text>
              </View>
              <Text style={styles.locationText}>
                📍 {result.morphology?.anatomical_location || result.quadrant || 'Cranial Center'}
              </Text>
            </View>

            {/* Layer Switching Tabs */}
            <Text style={styles.sectionHeader}>MULTI-SPECTRAL DIAGNOSTIC LAYERS</Text>
            <View style={styles.layerTabs}>
              {[
                { id: 'original', label: 'ORIGINAL' },
                { id: 'heatmap', label: 'GRAD-CAM' },
                { id: 'segmentation', label: 'MASK' },
                { id: 'contour', label: 'CONTOUR' },
                { id: 'bbox', label: 'ROI TARGET' },
              ].map((tab) => (
                <TouchableOpacity
                  key={tab.id}
                  style={[styles.layerTab, activeLayer === tab.id && styles.layerTabActive]}
                  onPress={() => {
                    Haptics.selectionAsync();
                    setActiveLayer(tab.id);
                  }}
                >
                  <Text style={[styles.layerTabText, activeLayer === tab.id && styles.layerTabTextActive]}>
                    {tab.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            {/* Quantitative Morphometric Biomarkers */}
            <Text style={styles.sectionHeader}>QUANTITATIVE BIOMARKERS</Text>
            <View style={styles.metricsGrid}>
              <View style={styles.metricCard}>
                <Text style={styles.metricLabel}>VOLUME BURDEN</Text>
                <Text style={[styles.metricVal, { color: '#00F0FF' }]}>
                  {result.morphology?.severity_pct || result.severity_pct || '0'}%
                </Text>
                <Text style={styles.metricSub}>{result.morphology?.severity_label || 'Intracranial'}</Text>
              </View>

              <View style={styles.metricCard}>
                <Text style={styles.metricLabel}>INFILTRATION INDEX</Text>
                <Text style={[styles.metricVal, { color: '#FFB800' }]}>
                  {result.morphology?.irregularity_score || '1.0'}
                </Text>
                <Text style={styles.metricSub}>{result.morphology?.infiltration_label || 'Isoperimetric'}</Text>
              </View>

              <View style={styles.metricCard}>
                <Text style={styles.metricLabel}>CLINICAL URGENCY</Text>
                <Text style={[styles.metricVal, { color: result.triage?.badge_color || '#22C55E' }]}>
                  {result.triage?.urgency || 'ROUTINE'}
                </Text>
                <Text style={styles.metricSub}>{result.triage?.who_grade || 'WHO Guidelines'}</Text>
              </View>

              <View style={styles.metricCard}>
                <Text style={styles.metricLabel}>LESION CENTROID</Text>
                <Text style={[styles.metricVal, { color: '#A855F7' }]}>
                  X:{result.morphology?.centroid?.x || '50'}%
                </Text>
                <Text style={styles.metricSub}>Y:{result.morphology?.centroid?.y || '50'}% Normalized</Text>
              </View>
            </View>

            {/* Clinical Action Plan */}
            <View style={styles.protocolCard}>
              <Text style={styles.protocolTitle}>RECOMMENDED CLINICAL ACTION PLAN</Text>
              <Text style={styles.protocolBody}>{result.triage?.action_protocol || result.impression}</Text>
            </View>

            {/* Reset / New Scan */}
            <TouchableOpacity 
              style={styles.newScanBtn} 
              onPress={() => {
                setResult(null);
                setSelectedImage(null);
              }}
            >
              <Text style={styles.newScanBtnText}>↺ SCAN ANOTHER PATIENT</Text>
            </TouchableOpacity>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#080C14',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: '#162338',
  },
  brandTitle: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '900',
    letterSpacing: 2,
  },
  brandAccent: {
    color: '#00F0FF',
  },
  brandSub: {
    color: '#556987',
    fontSize: 10,
    letterSpacing: 1.5,
    marginTop: 2,
    fontWeight: '700',
  },
  configBtn: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    backgroundColor: '#111C2D',
    borderWidth: 1,
    borderColor: '#1E324F',
  },
  configBtnText: {
    color: '#00F0FF',
    fontSize: 11,
    fontWeight: '700',
  },
  configBox: {
    backgroundColor: '#0F1A2A',
    padding: 14,
    borderBottomWidth: 1,
    borderBottomColor: '#1E324F',
  },
  configLabel: {
    color: '#8EA4C1',
    fontSize: 10,
    fontWeight: '700',
    marginBottom: 6,
    letterSpacing: 1,
  },
  configInput: {
    backgroundColor: '#080C14',
    borderColor: '#00F0FF',
    borderWidth: 1,
    borderRadius: 6,
    paddingHorizontal: 12,
    paddingVertical: 8,
    color: '#FFFFFF',
    fontSize: 13,
    fontFamily: 'monospace',
  },
  configHint: {
    color: '#556987',
    fontSize: 10,
    marginTop: 6,
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 40,
  },
  scanViewport: {
    width: '100%',
    aspectRatio: 1,
    backgroundColor: '#04070D',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#162338',
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative',
    overflow: 'hidden',
  },
  mriImage: {
    width: '100%',
    height: '100%',
  },
  emptyPrompt: {
    alignItems: 'center',
    padding: 20,
  },
  emptyReticle: {
    color: '#00F0FF',
    fontSize: 32,
    marginBottom: 10,
  },
  emptyTitle: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '800',
    letterSpacing: 1.5,
  },
  emptySubtitle: {
    color: '#556987',
    fontSize: 11,
    marginTop: 4,
    textAlign: 'center',
  },
  cornerBracket: {
    position: 'absolute',
    width: 20,
    height: 20,
    borderColor: '#00F0FF',
  },
  topLeft: { top: 10, left: 10, borderTopWidth: 2, borderLeftWidth: 2 },
  topRight: { top: 10, right: 10, borderTopWidth: 2, borderRightWidth: 2 },
  bottomLeft: { bottom: 10, left: 10, borderBottomWidth: 2, borderLeftWidth: 2 },
  bottomRight: { bottom: 10, right: 10, borderBottomWidth: 2, borderRightWidth: 2 },
  actionRow: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 16,
  },
  toolBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#0F1A2A',
    borderColor: '#1E324F',
    borderWidth: 1,
    paddingVertical: 12,
    borderRadius: 8,
    gap: 8,
  },
  toolBtnIcon: {
    fontSize: 16,
  },
  toolBtnText: {
    color: '#00F0FF',
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 1,
  },
  analyzeBtn: {
    backgroundColor: '#00F0FF',
    paddingVertical: 15,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 16,
    shadowColor: '#00F0FF',
    shadowOpacity: 0.4,
    shadowRadius: 10,
    elevation: 5,
  },
  analyzeBtnDisabled: {
    opacity: 0.6,
  },
  analyzeBtnText: {
    color: '#080C14',
    fontSize: 14,
    fontWeight: '900',
    letterSpacing: 1.5,
  },
  resultsContainer: {
    marginTop: 20,
  },
  diagCard: {
    padding: 16,
    borderRadius: 10,
    borderWidth: 1,
    marginBottom: 16,
  },
  cardHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  classBadge: {
    fontSize: 20,
    fontWeight: '900',
    letterSpacing: 1,
  },
  confBadge: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '700',
    fontFamily: 'monospace',
  },
  locationText: {
    color: '#D1D5DB',
    fontSize: 12,
    marginTop: 8,
  },
  sectionHeader: {
    color: '#556987',
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1.5,
    marginTop: 10,
    marginBottom: 8,
  },
  layerTabs: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 16,
  },
  layerTab: {
    backgroundColor: '#0E1726',
    borderWidth: 1,
    borderColor: '#1E324F',
    paddingHorizontal: 10,
    paddingVertical: 8,
    borderRadius: 6,
  },
  layerTabActive: {
    backgroundColor: '#00F0FF',
    borderColor: '#00F0FF',
  },
  layerTabText: {
    color: '#8EA4C1',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 1,
  },
  layerTabTextActive: {
    color: '#080C14',
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },
  metricCard: {
    flexBasis: '48%',
    flexGrow: 1,
    backgroundColor: '#0E1726',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#1E324F',
  },
  metricLabel: {
    color: '#556987',
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 1,
  },
  metricVal: {
    fontSize: 18,
    fontWeight: '900',
    marginTop: 4,
    fontFamily: 'monospace',
  },
  metricSub: {
    color: '#8EA4C1',
    fontSize: 10,
    marginTop: 2,
  },
  protocolCard: {
    backgroundColor: '#0A121E',
    borderColor: '#1E324F',
    borderWidth: 1,
    borderRadius: 8,
    padding: 14,
    marginBottom: 20,
  },
  protocolTitle: {
    color: '#00F0FF',
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1,
    marginBottom: 6,
  },
  protocolBody: {
    color: '#CBD5E1',
    fontSize: 12,
    lineHeight: 18,
  },
  newScanBtn: {
    borderWidth: 1,
    borderColor: '#556987',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  newScanBtnText: {
    color: '#8EA4C1',
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 1,
  },
});
