import React, { useState } from 'react';

function App() {
  const [activeTab, setActiveTab] = useState('video'); // 'video' or 'image'
  const [videoFile, setVideoFile] = useState(null);
  const [imageFile, setImageFile] = useState(null);

  const [dragActiveVideo, setDragActiveVideo] = useState(false);
  const [dragActiveImage, setDragActiveImage] = useState(false);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const BACKEND_URL = 'http://localhost:5000';

  // Drag and drop handlers
  const handleDrag = (e, setDragState) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragState(true);
    } else if (e.type === "dragleave") {
      setDragState(false);
    }
  };

  const handleDrop = (e, setFileState, setDragState, fileType) => {
    e.preventDefault();
    e.stopPropagation();
    setDragState(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (validateFile(file, fileType)) {
        setFileState(file);
      }
    }
  };

  const handleFileChange = (e, setFileState, fileType) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (validateFile(file, fileType)) {
        setFileState(file);
      }
    }
  };

  const validateFile = (file, type) => {
    setError(null);
    if (type === 'video' && !file.type.startsWith('video/')) {
      setError('Please upload a valid video file.');
      return false;
    }
    if (type === 'image' && !file.type.startsWith('image/')) {
      setError('Please upload a valid image file.');
      return false;
    }
    return true;
  };

  const handleClear = () => {
    setVideoFile(null);
    setImageFile(null);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (activeTab === 'video' && !videoFile) {
      setError('Please select a video file first.');
      return;
    }
    if (activeTab === 'image' && !imageFile) {
      setError('Please select a face image file first.');
      return;
    }

    setLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    if (activeTab === 'video') {
      formData.append('video', videoFile);
    } else {
      formData.append('image', imageFile);
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Inference failed on the server.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(err.message || 'An error occurred during prediction.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <header>
        <h1 id="app-title">Deepfake Detector</h1>
        <p className="subtitle">
          Detect synthetic faces and visual manipulations using CNN spatial models, Vision Transformers, and frequency analyses.
        </p>
      </header>

      <main className="glass-panel">
        <div className="tab-container" id="tab-switcher">
          <button
            id="tab-video"
            type="button"
            className={`tab-btn ${activeTab === 'video' ? 'active' : ''}`}
            onClick={() => { setActiveTab('video'); handleClear(); }}
          >
            Video Upload
          </button>
          <button
            id="tab-image"
            type="button"
            className={`tab-btn ${activeTab === 'image' ? 'active' : ''}`}
            onClick={() => { setActiveTab('image'); handleClear(); }}
          >
            Image Upload
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          {activeTab === 'video' ? (
            <div className="upload-grid">
              <div
                id="video-dropzone"
                className={`drop-zone ${dragActiveVideo ? 'drag-active' : ''}`}
                onDragEnter={(e) => handleDrag(e, setDragActiveVideo)}
                onDragOver={(e) => handleDrag(e, setDragActiveVideo)}
                onDragLeave={(e) => handleDrag(e, setDragActiveVideo)}
                onDrop={(e) => handleDrop(e, setVideoFile, setDragActiveVideo, 'video')}
                onClick={() => document.getElementById('video-input').click()}
              >
                <div className="upload-icon">📹</div>
                <h3>Drag & Drop Video</h3>
                <p style={{ color: 'var(--text-secondary)', marginTop: '0.25rem' }}>or click to browse local files</p>
                <input
                  id="video-input"
                  type="file"
                  accept="video/*"
                  style={{ display: 'none' }}
                  onChange={(e) => handleFileChange(e, setVideoFile, 'video')}
                />
                {videoFile && (
                  <div className="file-info" id="video-file-info">
                    Selected: <strong>{videoFile.name}</strong> ({(videoFile.size / (1024 * 1024)).toFixed(2)} MB)
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="upload-grid">
              <div
                id="image-dropzone"
                className={`drop-zone ${dragActiveImage ? 'drag-active' : ''}`}
                onDragEnter={(e) => handleDrag(e, setDragActiveImage)}
                onDragOver={(e) => handleDrag(e, setDragActiveImage)}
                onDragLeave={(e) => handleDrag(e, setDragActiveImage)}
                onDrop={(e) => handleDrop(e, setImageFile, setDragActiveImage, 'image')}
                onClick={() => document.getElementById('image-input').click()}
              >
                <div className="upload-icon">👤</div>
                <h3>Drag & Drop Face Crop</h3>
                <p style={{ color: 'var(--text-secondary)', marginTop: '0.25rem' }}>or click to browse image</p>
                <input
                  id="image-input"
                  type="file"
                  accept="image/*"
                  style={{ display: 'none' }}
                  onChange={(e) => handleFileChange(e, setImageFile, 'image')}
                />
                {imageFile && (
                  <div className="file-info" id="image-file-info">
                    Selected: <strong>{imageFile.name}</strong>
                  </div>
                )}
              </div>
            </div>
          )}

          {error && <div className="error-message" id="error-alert">{error}</div>}

          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
            {(videoFile || imageFile || result) && (
              <button
                id="btn-clear"
                type="button"
                className="btn-submit"
                style={{ background: 'transparent', border: '1px solid var(--border-color)', boxShadow: 'none' }}
                onClick={handleClear}
              >
                Clear
              </button>
            )}
            <button
              id="btn-submit-inference"
              type="submit"
              className="btn-submit"
              disabled={loading || (activeTab === 'video' ? !videoFile : !imageFile)}
            >
              {loading ? 'Running Detection...' : 'Scan for Deepfakes'}
            </button>
          </div>
        </form>

        {loading && (
          <div className="loader-container" id="loader">
            <div className="spinner"></div>
            <div className="loader-text">Analyzing visual spatial, semantic, and spectral features...</div>
          </div>
        )}

        {result && (
          <div className="results-container" id="results-display">
            <div className="results-header">
              <div
                id="result-badge"
                className={`result-badge ${result.prediction === 'REAL' ? 'real' : 'fake'}`}
              >
                {result.prediction}
              </div>
              <div className="confidence-info" id="confidence-text">
                Confidence Level: <span className="confidence-highlight">{result.confidence}%</span>
              </div>
            </div>

            <div className="stats-row">
              <div className="stat-card">
                <div className="stat-label">Real Signal Probability</div>
                <div className="stat-val real" id="real-probability-value">{result.real_prob}%</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Deepfake Manipulated Probability</div>
                <div className="stat-val fake" id="fake-probability-value">{result.fake_prob}%</div>
              </div>
            </div>

            <div className="heatmap-container" id="heatmap-card">
              <div className="heatmap-title">Grad-CAM Spatial Heatmap Overlay</div>
              <div className="heatmap-wrapper">
                <img
                  id="heatmap-img"
                  src={`${BACKEND_URL}${result.heatmap_url}?t=${Date.now()}`}
                  alt="Grad-CAM Deepfake Detection Heatmap Overlay"
                  className="heatmap-image"
                />
              </div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '1rem', textAlign: 'center', maxWidth: '600px' }}>
                Grad-CAM reveals which region of the facial crop features high-attention artifact concentrations that influenced the fusion classification network decision.
              </p>
            </div>
          </div>
        )}
      </main>

      <footer>
        <p>© 2026 Multi-Domain Deepfake Detection System. Built with PyTorch, Flask, and React.</p>
      </footer>
    </>
  );
}

export default App;
