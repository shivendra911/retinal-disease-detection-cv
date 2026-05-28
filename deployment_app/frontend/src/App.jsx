import { useState } from 'react'

function App() {
  const [formData, setFormData] = useState({
    Age: 55,
    BMI: 28.5,
    HbA1c: 6.8,
    Systolic_BP: 130,
    Diastolic_BP: 85,
    Cholesterol: 210
  });
  
  const [triageResult, setTriageResult] = useState(null);
  const [imageResult, setImageResult] = useState(null);
  const [loadingTriage, setLoadingTriage] = useState(false);
  const [loadingImage, setLoadingImage] = useState(false);

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: parseFloat(e.target.value) || 0
    });
  };

  const submitTriage = async (e) => {
    e.preventDefault();
    setLoadingTriage(true);
    setTriageResult(null);
    try {
      const response = await fetch('http://localhost:8000/api/triage', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      const data = await response.json();
      
      // Catch Pydantic validation errors
      if (response.status === 422) {
          setTriageResult({ error: "Input out of biological bounds. Please check your numbers." });
      } else {
          setTriageResult(data);
      }
    } catch (error) {
      console.error(error);
      setTriageResult({ error: "Failed to connect to backend service. Please check your network." });
    }
    setLoadingTriage(false);
  };

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // Client-side Edge Case: Not an image
    if (!file.type.startsWith('image/')) {
        setImageResult({ error: "Upload rejected: File must be an image format (JPEG, PNG, etc)." });
        return;
    }

    // Client-side Edge Case: Oversized file
    if (file.size > 15 * 1024 * 1024) {
        setImageResult({ error: "Upload rejected: Image exceeds the 15MB limit." });
        return;
    }
    
    setImageResult(null);
    setLoadingImage(true);
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      setImageResult(data);
    } catch (error) {
      console.error(error);
      setImageResult({ error: "Failed to connect to backend service." });
    }
    setLoadingImage(false);
  };

  const [selectedLanguage, setSelectedLanguage] = useState('english');
  const [reportResult, setReportResult] = useState(null);
  const [loadingReport, setLoadingReport] = useState(false);

  const generateReport = async (drGrade, hrPresent, riskScore) => {
    setLoadingReport(true);
    setReportResult(null);
    try {
      const response = await fetch('http://localhost:8000/api/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dr_grade: drGrade || 0,
          hr_present: hrPresent || 0.0,
          language: selectedLanguage,
          Age: formData.Age,
          BMI: formData.BMI,
          HbA1c: formData.HbA1c,
          Systolic_BP: formData.Systolic_BP,
          Diastolic_BP: formData.Diastolic_BP,
          Cholesterol: formData.Cholesterol,
          Risk_Score: riskScore
        })
      });
      const data = await response.json();
      setReportResult(data);
    } catch (error) {
      console.error(error);
      setReportResult({ status: 'error', message: "Failed to connect to LLM service." });
    }
    setLoadingReport(false);
  };

  return (
    <>
      <div className="header">
        <h1>IRDAS System</h1>
        <p>Intelligent Retinal Disease Assessment System</p>
      </div>

      <div className="dashboard">
        {/* Stage 1: Clinical Triage */}
        <div className="glass-card">
          <h2>Stage 1: Clinical Triage</h2>
          <form onSubmit={submitTriage}>
            <div className="form-group">
              <label>Age (0-120)</label>
              <input type="number" name="Age" min="0" max="120" value={formData.Age} onChange={handleInputChange} step="any" required />
            </div>
            <div className="form-group">
              <label>BMI (10-100)</label>
              <input type="number" name="BMI" min="10" max="100" value={formData.BMI} onChange={handleInputChange} step="any" required />
            </div>
            <div className="form-group">
              <label>HbA1c % (2-25)</label>
              <input type="number" name="HbA1c" min="2" max="25" value={formData.HbA1c} onChange={handleInputChange} step="any" required />
            </div>
            <div className="form-group">
              <label>Systolic BP (50-300)</label>
              <input type="number" name="Systolic_BP" min="50" max="300" value={formData.Systolic_BP} onChange={handleInputChange} step="any" required />
            </div>
            <div className="form-group">
              <label>Diastolic BP (30-200)</label>
              <input type="number" name="Diastolic_BP" min="30" max="200" value={formData.Diastolic_BP} onChange={handleInputChange} step="any" required />
            </div>
            <div className="form-group">
              <label>Cholesterol (50-600)</label>
              <input type="number" name="Cholesterol" min="50" max="600" value={formData.Cholesterol} onChange={handleInputChange} step="any" required />
            </div>
            <button type="submit" className="btn" disabled={loadingTriage}>
              {loadingTriage ? 'Analyzing...' : 'Assess Risk'}
            </button>
          </form>

          {triageResult && !triageResult.error && (
            <div className={`result-box ${triageResult.risk_category === 'High Risk' ? 'high-risk' : 'low-risk'}`}>
              <h3 style={{marginTop: 0}}>{triageResult.risk_category}</h3>
              <p>Risk Score: {(triageResult.risk_score * 100).toFixed(1)}%</p>
              <p style={{marginBottom: 0, opacity: 0.8}}>{triageResult.message}</p>
              
              <div style={{marginTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '1rem'}}>
                <label style={{display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', color: 'var(--text-muted)'}}>Patient Language:</label>
                <select 
                  value={selectedLanguage} 
                  onChange={(e) => setSelectedLanguage(e.target.value)}
                  style={{width: '100%', padding: '0.5rem', borderRadius: '4px', background: 'rgba(15,23,42,0.8)', color: 'white', border: '1px solid var(--border)', marginBottom: '1rem'}}
                >
                  <option value="english">English</option>
                  <option value="hindi">Hindi</option>
                  <option value="tamil">Tamil</option>
                  <option value="bengali">Bengali</option>
                </select>
                <button 
                  className="btn" 
                  style={{background: '#10B981'}} 
                  onClick={() => generateReport(0, triageResult.risk_score)}
                  disabled={loadingReport}
                >
                  {loadingReport ? 'Generating...' : 'Generate Patient Report'}
                </button>
              </div>
            </div>
          )}
          
          {triageResult?.error && (
            <div className="result-box high-risk">
              <p style={{margin: 0}}>{triageResult.error}</p>
            </div>
          )}
        </div>

        {/* Stage 2 & 3 */}
        <div>
          <div className="glass-card" style={{marginBottom: '2rem'}}>
            <h2>Stage 2: Fundus Grading</h2>
            <div 
              className="upload-zone" 
              style={{ 
                opacity: loadingImage ? 0.5 : 1, 
                pointerEvents: loadingImage ? 'none' : 'auto' 
              }}
              onClick={() => document.getElementById('file-upload').click()}
            >
              <p>{loadingImage ? '⚙️ Processing...' : '📁 Drag and drop or click to upload fundus image'}</p>
              <input 
                type="file" 
                id="file-upload" 
                style={{ display: 'none' }} 
                accept="image/*"
                onChange={handleImageUpload}
                disabled={loadingImage}
              />
            </div>
            {loadingImage && <p style={{textAlign: 'center', color: 'var(--text-muted)'}}>Running Deep Learning Pipeline...</p>}
            
            {imageResult && !imageResult.error && (
              <div className={`result-box ${imageResult.status === 'rejected' ? 'high-risk' : ''}`}>
                <h3 style={{marginTop: 0}}>Analysis Complete</h3>
                <p>Status: {imageResult.status}</p>
                <p style={{marginBottom: 0, opacity: 0.8}}>{imageResult.message}</p>
              </div>
            )}

            {imageResult?.error && (
              <div className="result-box high-risk">
                <p style={{margin: 0}}>{imageResult.error}</p>
              </div>
            )}
          </div>

          {/* Report Display */}
          {reportResult && (
            <div className="glass-card report-card">
              <h2>Stage 3: AI Patient Report</h2>
              {reportResult.status === 'error' ? (
                <div className="result-box high-risk"><p>{reportResult.message}</p></div>
              ) : (
                <div className="report-content">
                  <p style={{whiteSpace: 'pre-wrap', lineHeight: '1.6'}}>{reportResult.report}</p>
                </div>
              )}
            </div>
          )}
        </div>

      </div>
    </>
  )
}

export default App
