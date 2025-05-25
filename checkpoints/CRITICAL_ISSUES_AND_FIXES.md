# Critical Issues and Fixes - Custom Visualization Development
**Date:** 2025-05-25  
**Status:** BROKEN - needs immediate fixes

## Current Status: BROKEN
The latest "complete" visualization is not working. Issues include:
- Stuck at loading screen
- No data points visible
- Color legend not appearing
- JavaScript errors (likely)

## Root Causes of Failures

### 1. Three.js Version Conflicts
**Problem:** Mixed CDN sources and version mismatches
- Using `three.js/r128` but `OrbitControls` from different version
- CDN inconsistencies cause loading failures

**Solution:** Use consistent versions from same CDN

### 2. JavaScript Function Ordering Issues
**Problem:** Functions called before they're defined
- `createColorLegend()` called before `init()` completes
- Color functions used in geometry creation before definition

**Solution:** Proper function ordering and initialization sequence

### 3. Complex CSS Variable System
**Problem:** CSS custom properties not updating correctly
- Font size changes via CSS variables unreliable
- Background color transitions causing render issues

**Solution:** Direct style manipulation instead of CSS variables

### 4. Over-engineered Feature Creep
**Problem:** Added too many features without testing each incrementally
- Pan mode conflicts with raycasting
- Multiple control systems interfering
- Complex state management

**Solution:** Build minimal working version first, add features incrementally

## Specific Technical Issues

### Issue 1: SHAP Computation Blocking
```python
# PROBLEM: This blocks the main thread
shap_values = explainer.shap_values(X_res)  # Takes 30+ seconds
```
**Fix:** Pre-compute SHAP or use smaller sample for testing

### Issue 2: JSON Serialization of Large Data
```python
# PROBLEM: 9513 points × complex objects = massive JSON
data = {json.dumps(data_points)}  # Can exceed browser memory
```
**Fix:** Compress data or use binary format

### Issue 3: Three.js Geometry Updates
```python
# PROBLEM: Recreating BufferGeometry on filter changes
geometry.setAttribute('position', new THREE.BufferAttribute(...))
```
**Fix:** Pre-allocate larger buffers, show/hide points instead

### Issue 4: Event Handler Conflicts
```javascript
// PROBLEM: Multiple event handlers on same elements
addEventListener('mousemove', onMouseMove);  // Conflicts with pan mode
```
**Fix:** Single event dispatcher with mode checking

## Working vs Broken Versions

### ✅ WORKING: `umap_fixed_*_20250525_083528.html`
- Basic points rendering
- Filter checkboxes work
- Hover tooltips work
- Click to view essays work
- Simple controls

### ❌ BROKEN: `umap_complete_*_20250525_084541.html`
- Added too many features at once
- JavaScript errors prevent loading
- Complex CSS system fails
- Over-engineered controls

## Lessons Learned

### 1. Always Test Incrementally
- Each feature addition should be tested before adding next
- Keep working backup before major changes
- Use browser console to debug JavaScript errors

### 2. Three.js Best Practices
- Use consistent CDN sources
- Initialize in correct order: Scene → Camera → Renderer → Geometry → Controls
- Test geometry creation separately from interaction logic

### 3. Data Size Management
- 9513 points × rich metadata = performance issues
- Consider data pagination or LOD (Level of Detail)
- Pre-compute expensive operations offline

### 4. JavaScript Error Handling
```javascript
// ALWAYS wrap in try-catch for debugging
try {
    createPointCloud();
} catch (error) {
    console.error('Point cloud creation failed:', error);
}
```

## Immediate Action Plan

### Step 1: Start from Working Version
Use `scripts/create_custom_visualization_fixed.py` as base

### Step 2: Add Features One at a Time
1. Fix SHAP values (pre-compute offline)
2. Add color-coded backgrounds (simple version)
3. Add font controls (direct style manipulation)
4. Add pan mode (separate from rotation)
5. Add color legend (static HTML, not dynamic)

### Step 3: Debug Process
```bash
# Test each version in browser
1. Open browser console (F12)
2. Check for JavaScript errors
3. Verify data loading: console.log(data.length)
4. Test point rendering: scene.children.length
5. Verify raycasting: intersects.length
```

## Technical Recommendations for Future Agent

### 1. Use Minimal Three.js Setup
```javascript
// WORKING PATTERN:
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, width/height, 0.1, 2000);
const renderer = new THREE.WebGLRenderer();
const controls = new THREE.OrbitControls(camera, renderer.domElement);

// Create geometry BEFORE adding to scene
const geometry = new THREE.BufferGeometry();
const material = new THREE.PointsMaterial();
const points = new THREE.Points(geometry, material);
scene.add(points);

// THEN start animation loop
animate();
```

### 2. Data Processing Pipeline
```python
# 1. Load minimal data first
df_small = df.head(100)  # Test with 100 points

# 2. Pre-compute expensive operations
shap_values = compute_shap_offline(model, X_pca)  # Save to file

# 3. Create lightweight JSON
data_points = [{
    'x': float(x), 'y': float(y), 'z': float(z),
    'value': float(value), 'preview': text[:50]
}]
```

### 3. Feature Addition Order
1. Basic point rendering (no interactions)
2. Camera controls (orbit only)
3. Hover detection (console.log only)
4. Tooltip display (static position)
5. Click handling (console.log only)
6. Filter logic (array manipulation)
7. UI controls (one at a time)

### 4. Debug Helpers
```javascript
// Add debug output to HTML
function updateDebug(info) {
    document.getElementById('debug').innerHTML = info;
}

// Test each component
updateDebug(`Points: ${data.length}, Scene: ${scene.children.length}`);
```

## Files to Use/Avoid

### ✅ USE THESE:
- `scripts/create_custom_visualization_fixed.py` - Known working base
- Browser console for debugging
- Small test datasets (100-1000 points)

### ❌ AVOID THESE:
- `scripts/create_custom_visualization_complete.py` - Broken
- `umap_complete_*.html` files - Don't work
- Complex CSS variable systems
- Adding multiple features simultaneously

## Recovery Strategy

1. **Revert to working version**
2. **Add ONE feature at a time**
3. **Test in browser after each change**
4. **Use browser console to debug**
5. **Keep backups of working versions**

The key is: SIMPLE, INCREMENTAL, TESTED changes only.